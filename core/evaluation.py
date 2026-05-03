import time
import random

class ABTesting:
    def __init__(self):
        self.status = "inactive"

    def start_testing(self):
        self.status = "active"
        print("A/B testing started")

    def stop_testing(self):
        self.status = "inactive"
        print("A/B testing stopped")

    def get_status(self):
        return self.status


class DeepLearning:
    def __init__(self):
        self.status = "inactive"

    def start_learning(self):
        self.status = "active"
        print("Deep learning started")

    def stop_learning(self):
        self.status = "inactive"
        print("Deep learning stopped")

    def get_status(self):
        return self.status


class LiveStatus:
    def __init__(self):
        self.ab_testing = ABTesting()
        self.deep_learning = DeepLearning()

    def update_status(self):
        ab_status = self.ab_testing.get_status()
        dl_status = self.deep_learning.get_status()
        print(f"A/B testing status: {ab_status}, Deep learning status: {dl_status}")

    def start_ab_testing(self):
        self.ab_testing.start_testing()
        self.update_status()

    def stop_ab_testing(self):
        self.ab_testing.stop_testing()
        self.update_status()

    def start_deep_learning(self):
        self.deep_learning.start_learning()
        self.update_status()

    def stop_deep_learning(self):
        self.deep_learning.stop_learning()
        self.update_status()


def main():
    live_status = LiveStatus()
    live_status.start_ab_testing()
    time.sleep(2)
    live_status.start_deep_learning()
    time.sleep(2)
    live_status.stop_ab_testing()
    time.sleep(2)
    live_status.stop_deep_learning()


if __name__ == "__main__":
    main()
