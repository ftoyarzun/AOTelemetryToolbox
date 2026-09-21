import dao
import numpy as np
import astropy
import astroquery

from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

Simbad.add_votable_fields('V', 'R', 'J', 'H','ra', 'dec')
location = EarthLocation(lat=43.9308333*u.deg, lon=5.71333*u.deg, height=650*u.m)

result = Simbad.query_object("hip87585")

if result:
    print(result)
    ra_str = result['ra'][0]     # e.g., '18 36 56.336'
    dec_str = result['dec'][0]   # e.g., '+38 47 01.28'
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.deg, u.deg), frame='icrs')
    obstime = Time.now()
    altaz_frame = AltAz(obstime=obstime, location=location)
    altaz = coord.transform_to(altaz_frame)
    print(f"Altitude (elevation): {altaz.alt:.2f}")
    print(f"Azimuth: {altaz.az:.2f}")
else:
    print("Star not found.")



