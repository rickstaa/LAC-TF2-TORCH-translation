import timeit

# script settings
N_TIMES = int(1e10)
print("Test iadd performance vs normal addition...")

# First method to add int
setup_str = """
x = 0.0
"""
exec_str = """
x+=1.0
"""
time_1 = timeit.timeit(exec_str, setup=setup_str, number=N_TIMES)


# First method to add int
setup_2_str = """
x = 0.0
"""
exec_2_str = """
x+=1.0
"""
time_2 = timeit.timeit(exec_2_str, setup=setup_2_str, number=N_TIMES)

# Print result
print("\nTest iadd/normal add speed:")
print(f"- Iadd pass time: {time_1} s")
print(f"- Normal add pass time: {time_2} s")
