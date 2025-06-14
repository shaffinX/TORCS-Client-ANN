import msgParser
import carState
import carControl
import os
import re
import numpy as np
import tensorflow as tf
import pickle

class TorcsPredictor:
    def __init__(self, model_path='torcs_driver_model.keras', metadata_path='torcs_driver_metadata.pkl'):
        # Load metadata (feature columns, scaler, etc.)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)
        self.feature_columns = meta['feature_columns']
        self.scaler = meta['scaler']
        self.num_gears = meta['num_gears']

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    @staticmethod
    def process_focus_column(focus_str):
        if focus_str is None:
            return [-1.0] * 5
        try:
            values = [float(x) for x in str(focus_str).split()]
            if len(values) < 5:
                values.extend([-1.0] * (5 - len(values)))
            return values[:5]
        except:
            return [-1.0] * 5

    @staticmethod
    def process_opponent_data(data_str):
        if data_str is None:
            return [-1.0] * 4
        try:
            values = [float(x) for x in str(data_str).split()]
            if len(values) < 4:
                values.extend([-1.0] * (4 - len(values)))
            return values[:4]
        except:
            return [-1.0] * 4

    @staticmethod
    def parse_sensor_string(sensor_data_str):
        sensor_data = {}
        parts = re.findall(r'\(([^)]+)\)', sensor_data_str)
        for part in parts:
            if ' ' in part:
                key, values = part.split(' ', 1)
                try:
                    values = [float(v) for v in values.split()]
                    sensor_data[key] = values
                except:
                    sensor_data[key] = values
            else:
                sensor_data[part] = 0
        # Expand focus/opponent
        if 'focus' in sensor_data:
            focus_values = TorcsPredictor.process_focus_column(sensor_data['focus'])
            for i, val in enumerate(focus_values):
                sensor_data[f'focus_{i+1}'] = val
        for i in range(1, 6):
            opponent_key = f'opponent_{i}_data'
            if opponent_key in sensor_data:
                opp_values = TorcsPredictor.process_opponent_data(sensor_data[opponent_key])
                for j, metric in enumerate(['dist', 'speedX', 'speedY', 'speedZ']):
                    sensor_data[f'opponent_{i}_{metric}'] = opp_values[j]
        return sensor_data

    def build_feature_array(self, args_dict, sensor_data=None):
        feature_values = []
        for feature in self.feature_columns:
            if sensor_data and feature in sensor_data:
                val = sensor_data[feature]
                if isinstance(val, list):
                    feature_values.append(val[0] if val else 0)
                else:
                    feature_values.append(float(val))
            else:
                val = args_dict.get(feature, 0)
                feature_values.append(float(val) if val is not None else 0)
        return np.array([feature_values])

    @staticmethod
    def format_focus(focus):
        focus_values = [0.1] * 5
        focus_target = int(min(4, round(focus * 4)))
        focus_values[focus_target] = 0.9
        return " ".join([f"{v:.1f}" for v in focus_values])

    def predict(self, sensor_string=None, **features):
        """
        sensor_string: optional TORCS sensor input as a string (e.g., "(angle 0.01)...")
        features: keyword arguments for all numerical input features
        Returns: output string in TORCS format
        """
        if sensor_string:
            sensor_data = self.parse_sensor_string(sensor_string)
            arr = self.build_feature_array(features, sensor_data)
        else:
            arr = self.build_feature_array(features)
        input_scaled = self.scaler.transform(arr)
        preds = self.model.predict(input_scaled, verbose=0)
        accel = float(preds[0][0][0])
        brake = float(preds[1][0][0])
        gear = int(np.argmax(preds[2][0]))
        steer = float(preds[3][0][0])
        clutch = float(preds[4][0][0])
        focus = float(preds[5][0][0])
        meta = int(round(float(preds[6][0][0])))
        focus_str = self.format_focus(focus)
        return f"(accel {accel:.3f})(brake {brake:.3f})(gear {gear})(steer {steer:.3f})(clutch {clutch:.3f})(focus {focus_str})(meta {meta})"


class MLDriver(object):
    '''
    A ML-based driver object for the SCRC
    Using a TensorFlow model instead of rule-based decision making
    '''

    def __init__(self, stage, model_path='torcs_driver_model.keras', metadata_path='torcs_driver_metadata.pkl'):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage

        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Initialize ML model predictor
        self.predictor = TorcsPredictor(model_path, metadata_path)
        print(f"ML Driver initialized with model from {model_path}")

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for _ in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        """Use ML model to control the car based on sensor input"""
        self.state.setFromMsg(msg)
        
        # Get model prediction using the sensor data
        prediction = self.predictor.predict(sensor_string=msg)
        
        # Parse prediction and apply to car control
        parts = re.findall(r'\(([^)]+)\)', prediction)
        for part in parts:
            if ' ' in part:
                key, value = part.split(' ', 1)
                if key == 'accel':
                    self.control.setAccel(float(value))
                elif key == 'brake':
                    self.control.setBrake(float(value))
                elif key == 'gear':
                    self.control.setGear(int(float(value)))
                elif key == 'steer':
                    self.control.setSteer(float(value))
                elif key == 'clutch':
                    self.control.setClutch(float(value))
                elif key == 'meta':
                    self.control.setMeta(int(float(value)))
                elif key == 'focus':
                    # Handle focus if your control supports it
                    pass
        
        return self.control.toMsg()

    def onShutDown(self):
        """Called when shutting down"""
        pass

    def onRestart(self):
        """Called when restarting"""
        pass