"""access ztf dr3 internal data-release"""
import os
import warnings
import numpy as np
import pandas

IDR_PATH = os.getenv("ZTFIDRPATH")
if np.any([IDR_PATH.endswith(test_case) for test_case in ["/dr2", "/dr2/"]]):
    IDR_PATH = IDR_PATH.replace("/dr2", "").replace("//", "/")

# ============ #
# Top level    #
# ============ #
def get_data(release="dr3", which="salt2-T21", **kwargs):
    """Return the combined master list and SALT data for a given release.

    Parameters
    ----------
    release : str, optional
        The data release version (default is "dr3").

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the master list joined with SALT data,
        indexed by "ztfname".
    """
    # so far data is quite empty but more will come.
    masterlist = get_master_list(release=release)
    salt = get_saltdata(release=release, which=which, **kwargs)
    data = masterlist.join(salt)
    return data

def get_master_list(release="dr3"):
    """return the master list of ZTF objects"""
    return pandas.read_csv( os.path.join(IDR_PATH, release, "tables/object_lists/master_list.csv") ).set_index("ztfname")

def get_saltdata(release="dr3", which="salt2-T21", bands="gri", version=""):
    """Return the SALT data for a given release and parameters.

    Parameters
    ----------
    release : str, optional
        The data release version (default is "dr3").
    which : str, optional
        The SALT model to use (default is "salt2-T21").
    bands : str, optional
        The photometric bands to include (default is "gri").
    version : str, optional
        The version of the SALT data (default is "02092025").

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the SALT parameters for the specified release
        and parameters.
    """
    pathsalt_dir = os.path.join(IDR_PATH, release, "tables/saltfit")
    if version is not None and version != "":
        version = f"_{version}"
    else:
        version = ""
    basename = f"ztf{release}_{which}params_{bands}{version}.csv"
    return pandas.read_csv(os.path.join(pathsalt_dir, basename)).set_index("ztfname")

def get_target_saltdata(name, which="salt2-T21", **kwargs):
    """Return the SALT data for a target object.

    Parameters
    ----------
    name : str
        The name or identifier of the target object.
    which : str, optional
        The SALT model to use (default is "salt2-T21").
    **kwargs
        Additional keyword arguments passed to get_saltdata().

    Returns
    -------
    pandas.Series
        Series containing the SALT data for the specified target.
    """
    return get_saltdata(which=which, **kwargs).loc[name]

# ============ #
# Lightcurves  #
# ============ #
# from ztfidr3.io import IDR_PATH
def get_target_lightcurve(name, release="dr3", test_exist=True, load=True):
    """Get the target lightcurve for a ZTF object.

    Parameters
    ----------
    target : str
        The name or identifier of the target object.
    release : str, optional
        The data release version (default is "dr2").
    test_exist : bool, optional
        If True, check if the file exists before loading (default is True).
    load : bool, optional
        If True, load and return the data as a pandas DataFrame.
        If False, return only the file path (default is True).

    Returns
    -------
    pandas.DataFrame or str or None
        If load is True, returns a pandas DataFrame containing the lightcurve data.
        If load is False, returns the file path as a string.
        If test_exist is True and file does not exist, returns None.
    """
    fullpath = os.path.join(IDR_PATH, release, "lightcurves", f"{name}_LC.csv")
    if test_exist:
        if not os.path.isfile(fullpath):
            warnings.warn(f"No lc file for {name} ; {fullpath}")
            return None

    if not load:
        return fullpath

    return pandas.read_csv(fullpath, sep='\s+', comment='#')

# ============== #
# Classification #
# ============== #
def get_typing_table(release="dr3"):
    """ """
    fullpath = os.path.join(IDR_PATH, release, "tables/.dataset_creation", f"type_list.csv")
    typing_source = pandas.read_csv(fullpath, index_col=0).replace({"no": False, "yes": True})
    return typing_source

# ============ #
#   Spectra    #
# ============ #
def get_target_spectra(name, release="dr3"):
    """Return the spectra for a given target name.

    Parameters
    ----------
    name : str
        Target name
    release : str, optional
        Release version (default is "dr3")

    Returns
    -------
    list of Spectrum

    """
    specfile = get_spec_datafile(contains=name, release=release)
    if len(specfile) == 0:
        warnings.warn(f"No spectra found for {name}")
        return []

    from .spectrum import Spectrum
    return [Spectrum.from_filename(specfile_, release=release) for specfile_ in specfile["basename"]]

def parse_spectrum_file(filename):
    """Parse a spectrum file.

    Parse a ZTF spectrum file with format: name_mjd_instrument.ascii

    Parameters
    ----------
    filename : str
        Path to spectrum file

    Returns
    -------
    tuple
        Tuple of (name, mjd, instrument) extracted from the filename
    """
    import re
    # Parse filename to extract metadata
    basename = os.path.basename(filename)
    match = re.match(r"([^_]+)_([^_]+)_([^_]+)_([^.]+)\.DAT", basename)
    # targetname, nthobs, type, instrument.
    if not match:
        return "failed", 0, 0, "failed"
        #raise ValueError(f"Filename does not match expected format: {basename}")

    return match.groups()

def fetch_specfile(filename, release="dr3", accept_input_file=True):
    """Fetch the full path of a spectrum file from the IDR."""
    # If this path exist, it sounds correct.
    if os.path.exists(filename) and accept_input_file:
        return filename

    basename = os.path.basename(filename)
    if release == "dr3":
        filepath = os.path.join(IDR_PATH, release, "spectra", "standardised",  basename)
    else:
        filepath = os.path.join(IDR_PATH, release, "spectra", basename)

    return filepath

def get_spec_datafile(release="dr3", contains="*"):
    """Return the spectral data file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing spectral data with columns ["name", "mjd", "instrument"]
        and mjd cast as float type.
    """
    from glob import glob

    path_to_spec = os.path.join(IDR_PATH, release, "spectra", "standardised")
    all_spectra = glob(os.path.join(path_to_spec, f"{contains}*.DAT"))
    specdata = [parse_spectrum_file(filename) for filename in all_spectra]
    specdata = pandas.DataFrame(specdata, columns=["ztfname", "nthobs", "date", "instrument"])
    specdata["basename"] = [os.path.basename(filename) for filename in all_spectra]
    return specdata

# ============ #
#   HOST       #
# ============ #
def get_target_hostcutoutpath(name, release="dr3"):
    """Fetch the host cutout for a given target."""
    if release == "dr3":
        return os.path.join(IDR_PATH, release, "hosts/cutouts", f"{name}_hostcutout.jpg")
    else:
        raise NotImplementedError(f"Host cutout not available for release {release}")
