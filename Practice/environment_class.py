class Environment:
    def __init__(self, size):
        self.size = size
        self.left_limit = 0
        self.bottom_limit = 0
        self.right_limit = size
        self.top_limit = size