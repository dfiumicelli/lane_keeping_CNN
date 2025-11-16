#!/usr/bin/env python3

import os
from ament_index_python import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import time
from collections import deque
import onnxruntime as ort

class LaneKeepingCNN(Node):
    def __init__(self):
        super().__init__('lane_keeping_cnn')
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/image_rect/compressed', self.image_callback, 10)
        
        self.session = self._load_neural_network_onnx()
        self.nn_input_size = (320, 240)
        self.num_classes = 3
                
        # PID
        self.Kp = 2.0
        self.Ki = 0.1
        self.Kd = 0.4
        
        # Velocità
        self.base_linear_speed = 0.10
        self.max_linear_speed = 0.16
        self.min_linear_speed = 0.04
        self.max_angular_speed = 1.0
        
        # PID state
        self.error_history = deque(maxlen=10)
        self.previous_error = 0.0
        self.integral_error = 0.0
        
        # Velocity smoothing
        self.velocity_smooth_factor = 0.6
        self.previous_linear_velocity = 0.0
        self.previous_angular_velocity = 0.0
        
        # Lane detection
        self.roi_top = 0.15
        self.min_line_area = 200
        
        # Fallback
        self.fallback_mode = 'none'
        self.fallback_frames = 0
        self.max_missing_frames = 8
        self.left_missing = 0
        self.right_missing = 0
        
        # Frame rate
        self.min_frame_interval = 0.12
        self.last_command_time = time.time()
        
        # Debug
        self.debug_mode = True
        self.frame_count = 0

    def _load_neural_network_onnx(self):
        self.get_logger().info("Loading ONNX model...")
        try:
            try:
                # Prova prima il package 'line_follower_CNN'
                package_share_directory = get_package_share_directory('line_follower_CNN')
                model_path = os.path.join(package_share_directory, 'full_model_v4.onnx')
                self.get_logger().info(f"Cercando modello in package: {model_path}")
                
                if not os.path.exists(model_path):
                    # Se non esiste, prova nel package corrente
                    self.get_logger().warn(f"Modello non trovato in {model_path}")
                    current_package = 'line_follower_CNN'  # Cambia con il tuo package name
                    package_share_directory = get_package_share_directory(current_package)
                    model_path = os.path.join(package_share_directory, 'full_model_v4.onnx')
                    self.get_logger().info(f"Provando in: {model_path}")
                    
                    if not os.path.exists(model_path):
                        # Ulteriore fallback: cerca nella directory corrente
                        self.get_logger().warn(f"Modello non trovato, provando nella directory corrente")
                        if os.path.exists('full_model_v4.onnx'):
                            model_path = 'full_model_v4.onnx'
                        else:
                            raise FileNotFoundError(f"Modello ONNX non trovato in nessuna posizione!")
                
                self.get_logger().info(f"Modello trovato: {model_path}")
                
            except Exception as e:
                # Se get_package_share_directory fallisce, usa path relativo
                self.get_logger().warn(f"Errore package: {e}, usando path relativo")
                model_path = 'full_model_v4.onnx'
                
                if not os.path.exists(model_path):
                    # Prova dalla directory dello script
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(script_dir, 'full_model_v4.onnx')
                    self.get_logger().info(f"Cercando in directory dello script: {model_path}")
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            session = ort.InferenceSession(model_path, sess_options)
            self.get_logger().info("Model loaded successfully!")
            return session
            
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
            self.get_logger().error(f"Current working directory: {os.getcwd()}")
            self.get_logger().error(f"Files in current directory: {os.listdir('.')}")
            raise

    def image_callback(self, msg):
        try:
            current_time = time.time()
            if (current_time - self.last_command_time) < self.min_frame_interval:
                return
            
            self.frame_count += 1
            dt = current_time - self.last_command_time
            self.last_command_time = current_time
            if dt <= 0: dt = 0.033
            
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return
            
            height, width = cv_image.shape[:2]
            
            # NN prediction
            class_map = self._predict_lane_mask(cv_image)
            
            continua_mask = (class_map == 1).astype(np.uint8)
            discontinua_mask = (class_map == 2).astype(np.uint8)
            
            # Detect lines
            left_x, right_x = self._detect_lines_nn_fixed(continua_mask, discontinua_mask, width, height)
            
            # Determine mode
            mode = self._determine_mode(left_x, right_x)
            
            # Update missing counters
            self.left_missing = 0 if left_x is not None else self.left_missing + 1
            self.right_missing = 0 if right_x is not None else self.right_missing + 1
            
            # Fallback logic
            if self.left_missing > self.max_missing_frames or self.right_missing > self.max_missing_frames:
                self.fallback_mode = mode
                self.fallback_frames += 1
            else:
                self.fallback_mode = 'none'
                self.fallback_frames = 0
            
            # Calculate target
            target_x = self._calculate_target_fixed(mode, left_x, right_x, width)
            
            # Error calculation
            camera_center = width / 2.0
            error = (target_x - camera_center) / (width / 2.0)
            
            # PID
            pid_output = self._pid_simple(error, dt)
            
            # GENERATE COMMAND with INVERTED angular.z
            twist_cmd = self.generate_twist_fixed(pid_output, mode)
            self.cmd_vel_pub.publish(twist_cmd)
            
            # Debug
            if self.debug_mode and self.frame_count % 3 == 0:
                self.show_debug(cv_image, continua_mask, discontinua_mask, left_x, right_x, target_x, mode, error, camera_center, twist_cmd.angular.z)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}', exc_info=True)

    def _predict_lane_mask(self, image):
        """NN prediction"""
        height, width = image.shape[:2]
        
        image_resized = cv2.resize(image, (self.nn_input_size[0], self.nn_input_size[1]), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_norm = (image_rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges / edges.max()).astype(np.float32) if edges.max() > 0 else edges.astype(np.float32)
        
        image_4ch = np.concatenate([np.transpose(image_norm, (2, 0, 1)), edges[np.newaxis, :, :]], axis=0)
        image_tensor = np.expand_dims(image_4ch, 0).astype(np.float32)
        
        outputs = self.session.run(None, {'input': image_tensor})
        pred = outputs[0][0]
        class_map = np.argmax(pred, axis=0).astype(np.uint8)
        
        return cv2.resize(class_map, (width, height), interpolation=cv2.INTER_NEAREST)

    def _detect_lines_nn_fixed(self, continua_mask, discontinua_mask, width, height):
        """Detect lines from NN masks"""
        roi_top = int(height * self.roi_top)
        mid_x = width // 2
        
        left_x = None
        right_x = None
        
        # LEFT = Classe 2 (DISCONTINUA)
        left_roi = discontinua_mask[roi_top:, :mid_x]
        contours_left, _ = cv2.findContours(left_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_left) > 0:
            contour = max(contours_left, key=cv2.contourArea)
            if cv2.contourArea(contour) > self.min_line_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    left_x = int(M["m10"] / M["m00"])
        
        # RIGHT = Classe 1 (CONTINUA)
        right_roi = continua_mask[roi_top:, mid_x:]
        contours_right, _ = cv2.findContours(right_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_right) > 0:
            contour = max(contours_right, key=cv2.contourArea)
            if cv2.contourArea(contour) > self.min_line_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    right_x = int(M["m10"] / M["m00"]) + mid_x
        
        return left_x, right_x

    def _determine_mode(self, left_x, right_x):
        """Determine detection mode"""
        if left_x is not None and right_x is not None:
            return 'dual'
        elif right_x is not None:
            return 'right_only'
        elif left_x is not None:
            return 'left_only'
        else:
            return 'lost'

    def _calculate_target_fixed(self, mode, left_x, right_x, width):
        """Target calculation"""
        camera_center = width / 2.0
        
        if mode == 'dual':
            target_x = (left_x + right_x) / 2.0
            self.get_logger().debug(f"DUAL: Center = {target_x:.0f}")
        
        elif mode == 'left_only':
            # Solo discontinua (LEFT) → punta a DESTRA
            target_x = left_x + 280
            self.get_logger().info(f"LEFT_ONLY: Only discontinua at {left_x}, target RIGHT at {target_x:.0f}")
        
        elif mode == 'right_only':
            # Solo continua (RIGHT) → punta a SINISTRA
            target_x = right_x - 280
            self.get_logger().info(f"RIGHT_ONLY: Only continua at {right_x}, target LEFT at {target_x:.0f}")
        
        else:
            target_x = camera_center
        
        return np.clip(target_x, 50, width - 50)

    def _pid_simple(self, error, dt):
        """Simple PID control"""
        self.error_history.append(error)
        
        P = self.Kp * error
        
        self.integral_error += error * dt
        self.integral_error = max(-0.5, min(0.5, self.integral_error))
        I = self.Ki * self.integral_error
        
        if len(self.error_history) >= 2:
            derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
            D = self.Kd * derivative
        else:
            D = 0.0
        
        pid_output = P + I + D
        pid_output = max(-1.0, min(1.0, pid_output))
        
        self.previous_error = error
        return pid_output

    def generate_twist_fixed(self, pid_output, mode):
        """FIXED: Inverted angular.z sign"""
        twist = Twist()
        
        if mode == 'dual':
            speed_factor = 0.85
        elif mode in ['left_only', 'right_only']:
            speed_factor = 0.65
        else:
            speed_factor = 0.2
        
        error_factor = 1.0 - min(abs(pid_output), 0.6)
        final_speed = self.base_linear_speed * speed_factor * error_factor
        final_speed = max(self.min_linear_speed, min(self.max_linear_speed, final_speed))
        
        # INVERTED SIGN!
        # pid_output > 0 → target a DESTRA → angular.z NEGATIVO (gira destra)
        # pid_output < 0 → target a SINISTRA → angular.z POSITIVO (gira sinistra)
        target_angular = -pid_output * self.max_angular_speed
        
        twist.linear.x = (self.velocity_smooth_factor * self.previous_linear_velocity +
                         (1 - self.velocity_smooth_factor) * final_speed)
        twist.angular.z = (self.velocity_smooth_factor * self.previous_angular_velocity +
                          (1 - self.velocity_smooth_factor) * target_angular)
        
        self.previous_linear_velocity = twist.linear.x
        self.previous_angular_velocity = twist.angular.z
        
        return twist

    def show_debug(self, image, continua_mask, discontinua_mask, left_x, right_x, target_x, mode, error, camera_center, angular_z):
        """Debug visualization"""
        try:
            height, width = image.shape[:2]
            debug_img = image.copy()
            mid_x = width // 2
            
            steer_direction = "--> RIGHT" if error > 0 else ("<-- LEFT" if error < 0 else "→ CENTER")
            
            text = [
                f"ANGULAR FIXED: Mode: {mode} {steer_direction}",
                f"LEFT (Class 2/DISCONTINUA): {left_x if left_x else 'LOST!'} | RIGHT (Class 1/CONTINUA): {right_x if right_x else 'LOST!'}",
                f"Center: {camera_center:.0f} | Target: {target_x:.0f} | Error: {error:.3f}",
                f"Angular.z: {angular_z:.3f} | L_miss: {self.left_missing} | R_miss: {self.right_missing}",
            ]
            
            for i, txt in enumerate(text):
                color_text = (0, 0, 255) if "RIGHT" in txt or "LEFT" in txt or "LOST" in txt else (255, 255, 255)
                cv2.putText(debug_img, txt, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_text, 2)
            
            # Camera center line
            cv2.line(debug_img, (int(camera_center), 0), (int(camera_center), height), (0, 255, 255), 3)
            cv2.putText(debug_img, "CENTER", (int(camera_center) - 40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Lines detected
            if left_x is not None:
                cv2.circle(debug_img, (left_x, height//2), 10, (255, 165, 0), -1)
                cv2.putText(debug_img, "DISCONTINUA(2)", (left_x - 60, height//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            
            if right_x is not None:
                cv2.circle(debug_img, (right_x, height//2), 10, (0, 255, 0), -1)
                cv2.putText(debug_img, "CONTINUA(1)", (right_x - 50, height//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Target
            cv2.circle(debug_img, (int(target_x), height//2), 12, (0, 0, 255), 3)
            cv2.putText(debug_img, "TARGET", (int(target_x) - 35, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Arrow from center to target
            arrow_y = height // 2
            if target_x > camera_center:
                cv2.arrowedLine(debug_img, (int(camera_center), arrow_y), (int(target_x) - 15, arrow_y), (0, 0, 255), 4, tipLength=0.3)
            elif target_x < camera_center:
                cv2.arrowedLine(debug_img, (int(camera_center), arrow_y), (int(target_x) + 15, arrow_y), (0, 0, 255), 4, tipLength=0.3)
            
            # Masks
            combined_seg = np.zeros((height, width, 3), dtype=np.uint8)
            combined_seg[discontinua_mask > 0] = [255, 165, 0]
            combined_seg[continua_mask > 0] = [0, 255, 0]
            
            combined = np.hstack([debug_img, combined_seg])
            cv2.imshow('Correct steering direction', combined)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Debug error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LaneKeepingCNN()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
