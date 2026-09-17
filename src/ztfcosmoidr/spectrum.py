import os
import pandas
import numpy as np
import warnings

from .io import IDR_PATH

def read_standardized_specfile(filepath):
    """ Read spectroscopic data from a file.

    Parses a spectrum file with header information (lines starting with '#')
    and data columns. Header lines are expected to have 'key: value' format.

    Parameters
    ----------
    filepath : str
        Path to the spectrum file to read.

    Returns
    -------
    data : np.ndarray
        2D array of spectroscopic data (float type).
        lbda, flux[, ...]
    head_dict : dict
        Dictionary containing header information parsed from lines starting with '#'.
    """
    alldata = open(filepath).read().splitlines()
    # parsing the header, keeping the header key in as a dict
    head_data = [line for line in alldata if line.startswith("#")]
    header = {}
    for head_entry in head_data:
        head_entry = head_entry.replace("# ", "").split(":")
        if len(head_entry) != 2:
            continue # extra info
        header[head_entry[0].strip()] = head_entry[1].strip()

    # data parsing and building a dataframe
    data_in = [line for line in alldata if not line.startswith("#")]
    data = np.asarray([line.split() for line in data_in], dtype="float")
    data = pandas.DataFrame(data, columns=["lbda", "flux", "fluxerr_orig", "fluxerr"])
    data[data == -99] = np.nan

    return data, header

def fetch_snidresult_of_filename(filename, warn_if_notexist=True):
    """ Fetch the SNID result associated with a given spectrum filename.

    Parameters
    ----------
    filename : str
        Path to the spectrum file. The associated SNID result file is
        assumed to be named identically but with the extension replaced
        by '_snid.h5'.
    warn_if_notexist : bool, optional
        If True (default), a warning is raised if the SNID result file
        does not exist.

    Returns
    -------
    pysnid.snid.SNIDReader or None
        The SNID result reader object if the file exists, otherwise None.
    """
    extention = os.path.splitext(filename)[1]
    snidresult_file = filename.replace(f"{extention}","_snid.h5")
    if not os.path.isfile(snidresult_file):
        if warn_if_notexist:
            warnings.warn(f"snidres file does not exists {snidresult_file}")

        return None

    from pysnid.snid import SNIDReader
    return SNIDReader.from_filename(snidresult_file)


class Spectrum:
    def __init__(self, data, header, filename=None,
                snidresult=None):
        """Initialize a Spectrum object.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing spectroscopic data columns (lbda, flux, etc.).
        header : dict
            Dictionary containing header information from the spectrum file.
        filename : str, optional
            Path to the spectrum file.
        snidresult : pysnid.snid.SNIDReader, optional
            SNID result reader object associated with the spectrum, if available.
        """
        self._data = data
        self._header = header
        self._filename = filename
        self._snidresult = snidresult

    @classmethod
    def from_name(cls, name, release="dr3"):
        """Create a Spectrum object from a target name.

        Parameters
        ----------
        name : str
            Target name to locate the spectrum file.
        release : str, optional
            Release name to locate the file in the IDR_PATH directory structure.
            If provided, the file is searched in IDR_PATH/release/spectra/.

        Returns
        -------
        Spectrum
            A new Spectrum instance initialized with data and header from the file.

        Raises
        ------
        FileNotFoundError
            If the spectrum file for the given target name is not found.
        """
        from .io import get_spec_datafile
        specfile = get_spec_datafile(contains=name, release=release)
        if len(specfile) == 0:
            raise FileNotFoundError(f"No spectra found for {name}")

        # Assuming we take the first matching spectrum file for the target
        if len(specfile) == 1:
            return cls.from_filename(specfile.iloc[0]["basename"], release=release)
        else:
            warnings.warn(f"Multiple spectra found for {name}, returning a list of spectra")
            return [cls.from_filename(specfile_.basename, release=release) for specfile_ in specfile["basename"]]

    @classmethod
    def from_filename(cls, filename, release=None):
        """Create a Spectrum object from a spectrum file.

        Parameters
        ----------
        filename : str
            Path to the spectrum file or basename of the file.
        release : str, optional
            Release name to locate the file in the IDR_PATH directory structure.
            If provided, the file is searched in IDR_PATH/release/spectra/.
            Required if filename is not an absolute path that exists.

        Returns
        -------
        Spectrum
            A new Spectrum instance initialized with data and header from the file.

        Raises
        ------
        FileNotFoundError
            If the file is not found and no release is provided, or if the file
            is still not found after constructing the path with the release.
        ValueError
            If the data shape is not 2D with 2, 3, or 4 columns.
        """
        from .io import fetch_specfile
        # this file does not exist, maybe it is just the basename
        if not os.path.isfile(filename):
            if release is None:
                raise FileNotFoundError(f"File not found: {filename} and no release given, cannot fetch it.")

            filename = fetch_specfile(filename, release=release)
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File not found: {filename}")

        # snid file if any
        if filename is not None:
            snidresult = fetch_snidresult_of_filename(filename, warn_if_notexist=False)

        data, header = read_standardized_specfile(filename)
        return cls(data, header, snidresult=snidresult, filename=filename)


    # =============== #
    #   Methods       #
    # =============== #
    def get_phase(self, t0, redshift=None):
        """Calculate the phase of the observation.

        Parameters
        ----------
        t0 : float
            Reference time (e.g., time of explosion).
            phase is defined as time-t0.

        redshift : float, optional
            Redshift value to set the phase in rest-frame. If None
            phase is return in obs-frame

        Returns
        -------
        float
            Phase value (mjd - t0), or NaN if observation date is not available.
        """
        mjd = self.obsdate
        if mjd is None:
            return np.nan

        phase = mjd - t0
        if redshift is not None:
            phase /= (1+redshift)

        return phase

    # ------- #
    # PLOTS   #
    # ------- #
    def show(self, ax=None, **kwargs):
        """Plot the spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, a new figure and axes
            are created with figsize (7, 3).
        **kwargs
            Additional keyword arguments passed to ax.plot().

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7,3))
        else:
            fig = ax.figure

        line, *_ = ax.plot(self.lbda, self.flux, **kwargs)
        if self.variance is not None:
            err = self.error
            _ = ax.fill_between(self.lbda, self.flux-err, self.flux+err, alpha=0.3, color=line.get_color())

        ax.set_xlabel("wavelength", fontsize="large")
        ax.set_ylabel("flux", fontsize="large")
        return fig

    # =============== #
    #    Properties   #
    # =============== #
    @property
    def data(self):
        """Get the spectroscopic data.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the spectroscopic data (lbda, flux, variance, etc.).
        """
        return self._data

    @property
    def header(self):
        """Get the header information.

        Returns
        -------
        dict
            Dictionary containing metadata from the spectrum file.
        """
        return self._header

    @property
    def filename(self):
        """Get the filename of the spectrum.

        Returns
        -------
        str or None
            Path to the spectrum file, or None if not available.
        """
        return self._filename

    @property
    def snidresult(self):
        """Get the SNID result associated with the spectrum.

        Returns
        -------
        pysnid.snid.SNIDReader or None
            SNID result reader object if available, otherwise None.
        """
        return self._snidresult

    @property
    def lbda(self):
        """Get the wavelength array.

        Returns
        -------
        pandas.Series
            Wavelength values from the spectroscopic data.
        """
        return self.data["lbda"]

    @property
    def flux(self):
        """Get the flux array.

        Returns
        -------
        pandas.Series
            Flux values from the spectroscopic data.
        """
        return self.data["flux"]

    @property
    def error(self):
        """Get the variance array if available.

        Returns
        -------
        pandas.Series or None
            Variance values if present in the data, otherwise None.
        """
        return self.data.get("fluxerr", np.nan)

    @property
    def variance(self):
        """Get the variance array if available.

        Returns
        -------
        pandas.Series or None
            Variance values if present in the data, otherwise None.
        """
        return self.error**2

    @property
    def obsdate(self):
        """Get the observation date (MJD).

        Returns
        -------
        float or None
            Modified Julian Date (MJD) of the observation if available,
            otherwise None.
        """
        mjd = self.header.get("SPECTRUM_MJD", None)
        if mjd is not None:
            return float(mjd)

        return None

    @property
    def exptime(self):
        """Get the exposure time.

        Returns
        -------
        float or None
            Exposure time in seconds if available, otherwise None.
        """
        exposure = self.header.get("SPECTRUM_TEXPOSE", None)
        if exposure is not None:
            return float(exposure)

        return None

    @property
    def instrument(self):
        """Get the instrument name.

        Returns
        -------
        str
            Instrument identifier from the header, or 'unknown' if not specified.
        """
        return self.header.get("SPECTRUM_INSTRUMENT", "unknown")

    @property
    def telescope(self):
        """Get the telescope name.

        Returns
        -------
        str
            Telescope identifier from the header, or 'unknown' if not specified.
        """
        return self.header.get("SPECTRUM_TELESCOPE", "unknown")
