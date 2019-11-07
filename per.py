import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import numpy as np
import mixfit
import matplotlib.pyplot as plt
import json

center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(
    column_filters={'Bmag': '<13'},  # число больше — звёзд больше
    row_limit=10000
)
stars = vizier.query_region(
    center_coord,
    width=1.5 * u.deg,
    height=1.5 * u.deg,
    catalog='USNO-A2.0',
)[0]
ra = stars['RAJ2000']._data  # прямое восхождение, аналог долготы
dec = stars['DEJ2000']._data  # склонение, аналог широты
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi)
x2 = dec
x = np.vstack((x1, x2)).T
uniform_dens = 1/((np.max(x[:, 0])-np.min(x[:, 0]))*(np.max(x[:, 1])-np.min(x[:, 1])))
m = mixfit.em_double_cluster(x, uniform_dens, 0.2, np.array([-0.1, 56]), np.array([0.2, 0.2]), 0.3, np.array([0.1, 56]),
                             np.array([1, 0.4]))
plt.scatter(x[:, 0], x[:, 1])
plt.scatter(m[1][0], m[1][1], marker='o', s=70, label='$mu_1$')
plt.scatter(m[4][0], m[4][1], marker='o', s=70, label='$mu_2$')
plt.title('график рассеяния точек звёздного поля')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.savefig('per.png')
data = {
  "size_ratio": 1.5,
  "clusters": [
    {
      "center": {"ra": m[1][0]/(np.cos(m[1][1]/180 * np.pi)) + ra.mean(), "dec": m[1][1]},
      "sigma": np.sqrt(np.mean(m[2])),
      "tau": m[0]
    },
    {
      "center": {"ra": m[4][0]/(np.cos(m[4][1]/180 * np.pi)) + ra.mean(), "dec": m[4][1]},
      "sigma": np.sqrt(np.mean(m[5])),
      "tau": m[3]
    }
  ]
}
with open("per.json", "w") as f:
    json.dump(data, f)
