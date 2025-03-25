import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = {
    "X": [1.5]*19 + [1]*19 + [2]*19 + [0.5]*19,
    "Y": [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8] * 4,
    "B": [
        0.6, 3.5, 7.7, 13.8, 17.1, 16.9, 15.2, 15.5, 17.1, 18.3, 17, 15.7, 17, 19, 18.9, 13.9, 8.2, 3.9, 1.8,
        0.6, 3.6, 8.5, 14.2, 17.7, 17.6, 16, 17, 18.8, 19.6, 18.1, 16.5, 17.3, 19, 18.1, 12.4, 7.4, 3.7, 1,
        0.2, 1.8, 4.8, 8.8, 11.1, 10.8, 9.6, 9.5, 10.3, 10.6, 10, 9.4, 10, 11.2, 11, 8.5, 4.5, 2.1, 0.6,
        0.8, 3.4, 8.1, 13.3, 15.8, 15.5, 14.1, 15, 17, 17.5, 15.9, 14.5, 15.6, 16.8, 16, 11.2, 5.9, 2.6, 0.7
    ]
}

# faire une colormap du champs magnétique

df = pd.DataFrame(data)
pivot_table = df.pivot_table(values="B", index="Y", columns="X")
plt.pcolormesh(pivot_table, cmap="plasma", shading="auto")
plt.axis("equal")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.colorbar(label="B (mT)")
plt.title("Champs magnétique en fonction de la position")
plt.show() 




