import numpy as np

# Original values of tunes constants
# controller_gains = {
#     "kp": 8,
#     "ki": 2.5,
#     "kd": 1.5
# }

# Gains working better with DRL speed profile
controller_gains = {
    "kp": 6,
    "ki": 2,
    "kd": 2
}
print(controller_gains)
