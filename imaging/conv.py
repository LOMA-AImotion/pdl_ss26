import numpy as np
from scipy.ndimage import convolve, correlate

# Image I (4x4)
I = np.array([
    [ 7,  3,  6,  5],
    [ 2, 12,  8, 10],
    [ 8, 10, 11,  2],
    [ 6, 11,  4,  1]
])

# Filter H (2x2)
H = np.array([
    [ 2,  2],
    [ 0, -1]
])

# --- True convolution (scipy flips the kernel internally) ---
I_conv = convolve(I, H, mode='constant', cval=0)
print("Convolution output:")
print(I_conv)
print(f"I'(1,1) = {I_conv[1,1]}  ← should be 21\n")

# --- Cross-correlation (no flip) ---
I_corr = correlate(I, H, mode='constant', cval=0)
print("Cross-correlation output:")
print(I_corr)
print(f"Cross-corr at (1,1) = {I_corr[1,1]}\n")

H_flip = np.flip(H)  # vertical and horizontal mirroring

print(f"H_flip:\n{H_flip}\n")
# --- Cross-correlation (no flip) ---
I_corr = correlate(I, H_flip, mode='constant', cval=0)
print("Cross-correlation output with flipped filter:")
print(I_corr)