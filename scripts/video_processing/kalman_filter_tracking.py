import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, dt=1, process_noise=0.03):
        """
        Initialize a simple Kalman filter for tracking positions.

        Args:
        dt (float): Time step between measurements.
        process_noise (float): Process noise parameter.
        """
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32) * process_noise

    def predict(self):
        """Predict the next state."""
        return self.kf.predict()

    def correct(self, measurement):
        """
        Correct the state of the Kalman filter.

        Args:
        measurement (list): Measured [x, y] coordinates of the object.
        
        Returns:
        numpy.ndarray: The corrected state.
        """
        return self.kf.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))

if __name__ == "__main__":
    # Example usage
    kf = KalmanFilter()
    current_measurement = [150, 100]
    kf.correct(current_measurement)
    for _ in range(5):
        print("Predicted state:", kf.predict())
