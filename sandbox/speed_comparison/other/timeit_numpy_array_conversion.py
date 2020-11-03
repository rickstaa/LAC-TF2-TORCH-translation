"""Small script to see which method of creating a numpy array, when input type is
unknown is faster.

Conclusions:
    - The manual method 1 is a lot faster than the automatic version.
    - The isinstance method is faster than the type comparison.
"""

import timeit

# Script settings
N_SAMPLE = int(5e8)

# Test converted version
setup_code = """
import numpy as np
test_array = np.random.rand(30,60)
"""
code = """
test_array
"""
method_1_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Test automatic converted function
setup_code = """
import numpy as np
test_array = np.random.rand(30,60)
"""
code = """
np.array(test_array)
"""
method_2_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Test manual converted function
setup_code = """
import numpy as np
test_array = np.random.rand(30,60)
"""
code = """
test_array if isinstance(test_array, (np.ndarray, list)) else np.array(test_array)
"""
method_3_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Test manual converted function 2
setup_code = """
import numpy as np
test_array = np.random.rand(30,60)
"""
code = """
test_array if type(test_array) in (np.ndarray, list) else np.array(test_array)
"""
method_4_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)

# Test manual converted function 3
setup_code = """
import numpy as np
test_array = np.random.rand(30,60)
"""
code = """
test_array if isinstance(test_array, (np.ndarray, list)) else [r]
"""
method_5_time = timeit.timeit(code, setup=setup_code, number=N_SAMPLE)


# Print results
print("\nTest numpy conversion methods:")
print(f"- No conversion: {method_1_time} s")
print(f"- Automatic conversion: {method_2_time} s")
print(f"- Manual conversion 1: {method_3_time} s")
print(f"- Manual conversion 2: {method_4_time} s")
print(f"- Manual conversion 3: {method_5_time} s")
