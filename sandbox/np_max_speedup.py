"""Test what is the fastest way to make sure numpy array values are positive.
"""

import numpy as np
import timeit

# Create dummy array
# sampl = np.random.uniform(low=-50, high=50, size=(50,))
sampl = np.array([-3.0])

# = Clip values to be positive

# Method 1
time1 = timeit.timeit(
    """
import numpy as np
sampl = np.array([-3.0])
max_val = np.max([sampl, np.zeros([1])])
""",
    number=100000,
)
time2 = timeit.timeit(
    """
import numpy as np
sampl = np.array([-3.0])
max_val = np.max(np.append(sampl, 0.0))
""",
    number=100000,
)
time3 = timeit.timeit(
    """
import numpy as np
sampl = np.array([-3.0])
max_val = np.clip(sampl, a_min=0.0, a_max=np.inf)
""",
    number=100000,
)
time4 = timeit.timeit(
    """
import numpy as np
sampl = np.array([-3.0])
max_val = sampl if sampl >= 0.0 else np.array([0.0])
""",
    number=100000,
)


# Print results
print(f"method 1: {time1}")
print(f"method 2: {time2}")
print(f"method 2: {time3}")
print(f"method 3: {time4}")
print("end")
