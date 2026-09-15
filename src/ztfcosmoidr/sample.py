import warnings

import pandas
import numpy as np
from . import io



class Sample():
    """
    A class to manage sample data and provide access to lightcurves and spectra.

    This class serves as a container for sample data from a specific ZTF data release,
    providing methods to retrieve lightcurves and spectra for individual targets.
    """
    def __init__(self, data, release="dr3"):
        """
        Initialize a Sample instance.

        Parameters
        ----------
        data : pandas.DataFrame
            The sample data containing target information.
        release : str, optional
            The ZTF data release version (default is "dr3").
        """
        self._data = data
        self._release = release
        # this is store for speed
        self._specfile = io.get_spec_datafile(release=release).set_index("ztfname")

    @classmethod
    def load_release(cls, release):
        """
        Load sample data from a specific ZTF data release.

        Parameters
        ----------
        release : str
            The ZTF data release version to load.

        Returns
        -------
        Sample
            A new Sample instance initialized with data from the specified release.
        """
        data = io.get_data(release=release)
        return cls(data)

    # ================ #
    #   Methods        #
    # ================ #
    def get_missing_classication(self, max_redshift=0.065):
        """ """
        return self.typinglist[(~self.typinglist["in_DR2"]) &
                                (self.typinglist["redshift"]<=max_redshift) &
                                (~self.typinglist["only_1_spec_after_7d"])]

    def get_target_lightcurve(self, name, as_dataframe=False):
        """
        Retrieve the lightcurve for a specific target.

        Parameters
        ----------
        name : str
            The name/identifier of the target.
        as_dataframe : bool, optional
            If True, return the raw data as a DataFrame. If False, return a
            LightCurve object (default is False).

        Returns
        -------
        pandas.DataFrame or LightCurve
            The lightcurve data for the target. Returns DataFrame if as_dataframe=True,
            otherwise returns a LightCurve object.

        Raises
        ------
        ValueError
            If the target is not found in the sample data.
        """
        if name not in self.data.index:
            raise ValueError(f"Target {name} not found in data")

        data = io.get_target_lightcurve(name, release=self.release)
        if as_dataframe:
            return data

        from .lightcurve import LightCurve
        saltdata = self.data.loc[name] # actually more than just salt data.
        return LightCurve(data=data, saltdata=saltdata)

    def get_target_spectra(self, name):
        """
        Retrieve the spectrum or spectra for a specific target.

        Parameters
        ----------
        name : str
            The name/identifier of the target.

        Returns
        -------
        list of Spectrum
            A list of Spectrum objects for the target. This can be a list of
            just one Spectrum, or an empty list if no spectrum is available.

        Raises
        ------
        ValueError
            If the target is not found in the sample data.
        """
        from .spectrum import Spectrum
        if name not in self.data.index:
            raise ValueError(f"Target {name} not found in data")

        if name not in self.specfile.index:
            warnings.warn(f"Target {name} has no spectrum")
            return []

        entry = self.specfile.loc[name]

        if isinstance(entry, pandas.Series):
            spec = [Spectrum.from_filename(entry.basename, release=self.release)]
        else:
            spec = [Spectrum.from_filename(basename, release=self.release) for basename in entry["basename"].values]

        return spec

    # ------- #
    # PLOTS   #
    # ------- #
    def show_target_lightcurve(self, name, ax=None, **kwargs):
        """
        Display the lightcurve for a specific target.

        Parameters
        ----------
        name : str
            The name/identifier of the target.
        ax : matplotlib.axes.Axes, optional
            A matplotlib Axes object on which to plot. If None, a new figure
            and axes will be created (default is None).
        **kwargs
            Additional keyword arguments to pass to the lightcurve plotting method.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        lc = self.get_target_lightcurve(name, as_dataframe=False)
        return lc.show(ax=ax, **kwargs)

    def show_target_spectra(self, name, axes=None, **kwargs):
        """
        Display the spectrum or spectra for a specific target.

        Parameters
        ----------
        name : str
            The name/identifier of the target.
        axes : list of matplotlib.axes.Axes, optional
            A list of matplotlib Axes objects on which to plot the spectra.
            If None, new figures and axes will be created (default is None).
        **kwargs
            Additional keyword arguments to pass to the spectrum plotting method.

        Returns
        -------
        list of matplotlib.axes.Axes
            A list of axes objects containing the plots.

        Raises
        ------
        ValueError
            If the target is not found in the sample data.
        """
        spectra = self.get_target_spectra(name)
        if axes is not None:
            if len(spectra)==len(axes):
                warnings.warn(f"Number of spectra ({len(spectra)}) does not match number of axes ({len(axes)})")

        return [spec.show(ax=ax_, **kwargs) for spec, ax_ in zip(spectra, axes)]

    def show_target_hostcutout(self, name, ax=None, **kwargs):
        """ Show the host cutout for a target.

        Parameters
        ----------
        name : str
            The name of the target.
        ax : matplotlib.axes.Axes, optional
            The axes object on which to plot the host cutout.
            If None, a new figure and axes will be created (default is None).
        **kwargs
            Additional keyword arguments to pass to the host cutout plotting method.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the host cutout plot.
        """
        from .hosts import get_target_cutoutimg
        return get_target_cutoutimg(name, ax=ax, release=self.release, **kwargs)

    # =============== #
    #   Properties    #
    # =============== #
    @property
    def data(self):
        """
        Get the sample data.

        Returns
        -------
        pandas.DataFrame
            The sample data containing target information.
        """
        return self._data

    @property
    def release(self):
        """
        Get the ZTF data release version.

        Returns
        -------
        str
            The ZTF data release version used by this sample.
        """
        return self._release

    @property
    def specfile(self):
        """
        Get the spectrum file information.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing spectrum file metadata indexed by target name.
        """
        return self._specfile

    @property
    def typinglist(self):
        """
        Get the list of classifications for targets in the sample

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the typing list for the sample.
        """
        if not hasattr(self, "_typinglist"):
            self._typinglist = io.get_typing_table(release=self.release).loc[self.data.index]

        return self._typinglist
