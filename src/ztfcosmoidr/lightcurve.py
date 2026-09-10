""" module focused on the LightCurve object """
import warnings

import pandas
import numpy as np

ZTFCOLOR = { # ZTF
        "ztfr":dict(marker="o", ms=7, mfc="C3"),
        "ztfg":dict(marker="o", ms=7, mfc="C2"),
        "ztfi":dict(marker="o", ms=7, mfc="C1")
}

BAD_ZTFCOLOR = { # ZTF
        "ztfr":dict(marker="o", ms=6, mfc="None", mec="C3"),
        "ztfg":dict(marker="o", ms=6, mfc="None", mec="C2"),
        "ztfi":dict(marker="o", ms=6, mfc="None", mec="C1")
}

def get_saltmodel(which="salt2.4-T21", **params):
    """Get a SALT2 model with specified parameters.

    Parameters
    ----------
    which : str, optional
        Model specification string. Can include version with "v=" syntax.
        Default is "salt2.4-T21".
    **params : dict
        Additional parameters to set on the model.

    Returns
    -------
    sncosmo.Model
        A SALT2 model with Milky Way dust effects included.

    Examples
    --------
    >>> model = get_saltmodel()
    >>> model = get_saltmodel("salt2.4", z=0.5, t0=55000)
    """
    import sncosmo
    if which is None or which in ["salt2.4"]:
        which = "salt2 v=2.4"
    if "salt" in which.lower() and "t21" in which.lower():
        which = "salt2 v=T21"

    # parsing model version
    source_name, *version = which.split("v=")
    source_name = source_name.strip()
    version = None if len(version)==0 else version[0].strip()
    source = sncosmo.get_source(source_name, version=version, copy=True)

    dust  = sncosmo.CCM89Dust()
    model = sncosmo.Model(source, effects=[dust],
                              effect_names=['mw'],
                              effect_frames=['obs'])
    model.set(**params)
    return model


class LightCurve():
    def __init__(self, data, saltdata=None):
        self._data = data
        self._saltdata = saltdata

    @classmethod
    def from_name(cls, name, release="dr3", **kwargs):
        """Create a LightCurve instance from a target name.

        Parameters
        ----------
        name : str
            The name of the target.
        release : str, optional
            Data release version. Default is "dr3".
        **kwargs : dict
            Additional keyword arguments to pass to get_target_saltdata.

        Returns
        -------
        LightCurve
            A new LightCurve instance with data and salt data loaded.
        """
        from . import io
        data = io.get_target_lightcurve(name, release=release)
        saltdata = io.get_target_saltdata(name, release=release, **kwargs)
        return cls(data, saltdata=saltdata)

    @classmethod
    def from_filename(cls, filename, saltdata=None):
        """Create a LightCurve instance from a data file.

        Parameters
        ----------
        filename : str
            Path to the light curve data file in whitespace-separated format.
        saltdata : pandas.DataFrame, optional
            SALT2 parameter data. Default is None.

        Returns
        -------
        LightCurve
            A new LightCurve instance with loaded data.
        """
        data = pandas.read_csv(filename, sep=r'\s+', comment='#')
        return cls(data, saltdata=saltdata)

    # =============== #
    #  Methods        #
    # =============== #
    def get_saltmodel(self, saltdata=None, **kwargs):
        """Get a SALT2 model with parameters from salt data.

        Parameters
        ----------
        saltdata : pandas.DataFrame, optional
            SALT2 parameter data. If None, uses the instance's saltdata.
        **kwargs : dict
            Additional keyword arguments to pass to get_saltmodel.

        Returns
        -------
        sncosmo.Model
            A SALT2 model with parameters set from saltdata.

        Raises
        ------
        ValueError
            If saltdata is not provided and not set on the instance.
        """

        if saltdata is None:
            saltdata = self.saltdata

        if saltdata is None:
            raise ValueError("saltdata is not given nor set")

        propmodel = saltdata[["z","t0","x0","x1","c","mwebv"]].to_dict()
        return get_saltmodel(**(propmodel | kwargs) )

    def get_lcdata(self, zp=30, mjdrange=None,
                       min_detection=None,
                       filters=None,
                       flagout=[1,2,4,8,16]):
        """Get light curve data with optional filtering and corrections.

        Parameters
        ----------
        zp : float, optional
            Zero point for flux calibration. If None, uses the data's ZP values.
        mjdrange : tuple of float, optional
            MJD range (min, max) to filter the light curve data.
        min_detection : float, optional
            Minimum detection significance (flux/flux_err) threshold.
        filters : str, list of str, or None, optional
            Filter selection. Options are:
            - None, '*', or 'all': no filter selection
            - str: single filter (e.g. 'ztfg')
            - list of str: multiple filters (e.g. ['ztfg', 'ztfr'])
            Default is None.
        flagout : list of int or str, optional
            Quality flags to exclude. Default is [1, 2, 4, 8, 16].
            flag == 0 means all good. Flag meanings:
            - 0: no warning
            - 1: flux_err==0, remove unphysical errors
            - 2: chi2dof>3, remove extreme outliers
            - 4: cloudy>1, BTS cut
            - 8: infobits>0, BTS cut
            - 16: mag_lim<19.3, cut applied in Dhawan 2021
            - 32: seeing>3, cut applied in Dhawan 2021
            - 64: fieldid>879, recommended IPAC cut
            - 128: moonilf>0.5, recommended IPAC cut
            - 256: has_baseline>1, has valid baseline correction
            - 512: airmass>2, recommended IPAC cut
            - 1024: flux/flux_err>=5, nominal detection

        Returns
        -------
        pandas.DataFrame
            Filtered light curve data with columns: mjd, mag, mag_err, filter,
            field_id, flag, rcid, zp, flux, error, detection, mag_lim, and phase.

        """

        from .utils import flux_to_mag

        if flagout in ["all","any","*"]:
            data = self.data[self.data["flag"]==0]
        elif flagout is None:
            data = self.data.copy()
        else:
            flag_ = np.all([(self.data.flag&i_==0) for i_ in np.atleast_1d(flagout)], axis=0)
            data = self.data[flag_]

        if zp is None:
            zeropoint = data["ZP"].values
            coef = 1.
        else:
            coef = 10 ** (-(data["ZP"].values - zp) / 2.5)
            zeropoint = np.ones(coef.shape) * zp

        flux  = data["flux"] * coef
        error = data["flux_err"] * coef
        detection = flux/error

        lcdata = data[["mjd","mag","mag_err","filter","field_id", "flag", "rcid"]] # "x_pos","y_pos"
        additional = pandas.DataFrame(np.asarray([zeropoint, flux, error, detection]).T,
                                         columns=["zp", "flux", "error", "detection"],
                                         index=lcdata.index)

        additional["mag_lim"], _ = flux_to_mag(error*5, None, zp=zp)

        lcdata = pandas.merge(lcdata, additional, left_index=True, right_index=True)

        if self.saltdata is not None:
            lcdata["phase"] = lcdata["mjd"]-self.saltdata['t0']
        else:
            lcdata["phase"] = np.NaN

        if mjdrange is not None:
            lcdata = lcdata[lcdata["mjd"].between(*mjdrange)]

        if min_detection is not None:
            lcdata = lcdata[lcdata["detection"]>min_detection]

        if filters is not None and filters not in ["*","all"]:
            lcdata = lcdata[lcdata["filter"].isin(np.atleast_1d(filters))]

        return lcdata

    def show(self, ax=None, zp=30, bands="*",
                formattime=True, zeroline=True,
                incl_salt=True, which_model=None, autoscale_salt=True, clear_yticks=True,
                phase_range=[-30,100], as_phase=False,
                zprop={}, inmag=False, ulength=0.1, ualpha=0.1,
                rm_flags=True, **kwargs):
        """Display the light curve with optional SALT2 model overlay.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object. If None, a new figure and axes are created.
        zp : float, optional
            Zero point for flux calibration. If None, uses the data's ZP values.
        bands : str, list of str, or None, optional
            Filter selection. Options are:
            - None, '*', or 'all': show all bands
            - str: single filter (e.g. 'ztfg')
            - list of str: multiple filters
            Default is "*".
        formattime : bool, optional
            Format x-axis as dates. Default is True.
        zeroline : bool, optional
            Draw a zero line on the plot. Default is True.
        incl_salt : bool, optional
            Include SALT2 model in the plot. Default is True.
        which_model : str, optional
            SALT2 model specification. Default is None (uses default).
        autoscale_salt : bool, optional
            Automatically scale axes to fit the model. Default is True.
        clear_yticks : bool, optional
            Clear y-axis tick labels. Default is True.
        phase_range : list of float, optional
            Phase range [min, max] in days relative to t0. Default is [-30, 100].
        as_phase : bool, optional
            Plot time as phase (days from t0) instead of MJD. Default is False.
        zprop : dict, optional
            Additional properties for the zero line. Default is {}.
        inmag : bool, optional
            Plot in magnitude instead of flux. Default is False.
        ulength : float, optional
            Upper limit arrow length. Default is 0.1.
        ualpha : float, optional
            Upper limit arrow transparency. Default is 0.1.
        notplt : bool, optional
            Do not plot (reserved for future use). Default is False.
        rm_flags : bool, optional
            Remove flagged data points. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to errorbar function.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plot.

        Notes
        -----
        The plot displays light curve data with error bars, color-coded by filter.
        Good data points are filled, bad data points are outlined. SALT2 model
        light curves can be overlaid if incl_salt is True.
        """
        passed_kwargs = {key:val for key,val in locals().items()
                         if key not in ["self", "incl_salt", "kwargs"]}
        if incl_salt:
            saltmodel = self.get_saltmodel(which=which_model)
        else:
            saltmodel = None

        t0 = self.saltdata.t0
        lcdata = self.get_lcdata(zp=zp)

        return self.__class__._show(lcdata, t0=t0, saltmodel=saltmodel,
                                    **(passed_kwargs | kwargs))

    @staticmethod
    def _show(lcdata, t0, saltmodel=None,
                ax=None, zp=30, bands="*",
                formattime=True, zeroline=True,
                autoscale_salt=True, clear_yticks=True,
                phase_range=[-30,100], as_phase=False,
                zprop={}, inmag=False, ulength=0.1, ualpha=0.1,
                rm_flags=True, **kwargs):
        """Display the light curve with optional SALT2 model overlay.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object. If None, a new figure and axes are created.
        figsize : tuple of float, optional
            Figure size (width, height). Default is None.
        zp : float, optional
            Zero point for flux calibration. If None, uses the data's ZP values.
        bands : str, list of str, or None, optional
            Filter selection. Options are:
            - None, '*', or 'all': show all bands
            - str: single filter (e.g. 'ztfg')
            - list of str: multiple filters
            Default is "*".
        formattime : bool, optional
            Format x-axis as dates. Default is True.
        zeroline : bool, optional
            Draw a zero line on the plot. Default is True.
        which_model : str, optional
            SALT2 model specification. Default is None (uses default).
        autoscale_salt : bool, optional
            Automatically scale axes to fit the model. Default is True.
        clear_yticks : bool, optional
            Clear y-axis tick labels. Default is True.
        phase_range : list of float, optional
            Phase range [min, max] in days relative to t0. Default is [-30, 100].
        as_phase : bool, optional
            Plot time as phase (days from t0) instead of MJD. Default is False.
        t0 : float, optional
            Explosion time. If None, uses t0 from saltdata. Default is None.
        zprop : dict, optional
            Additional properties for the zero line. Default is {}.
        inmag : bool, optional
            Plot in magnitude instead of flux. Default is False.
        ulength : float, optional
            Upper limit arrow length. Default is 0.1.
        ualpha : float, optional
            Upper limit arrow transparency. Default is 0.1.
        notplt : bool, optional
            Do not plot (reserved for future use). Default is False.
        rm_flags : bool, optional
            Remove flagged data points. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to errorbar function.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plot.

        Notes
        -----
        The plot displays light curve data with error bars, color-coded by filter.
        Good data points are filled, bad data points are outlined. SALT2 model
        light curves can be overlaid if incl_salt is True.
        """
        from matplotlib import dates as mdates
        from astropy.time import Time

        # - Axes Definition
        if ax is None:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[7,4])# if figsize is None else figsize)
            ax = fig.add_axes([0.1,0.15,0.8,0.75])
        else:
            fig = ax.figure

        # - End axes definition
        # --
        # - Data
        base_prop = dict(ls="None", mec="0.9", mew=0.5, ecolor="0.7", zorder=7)
        bad_prop  = dict(ls="None", mew=1, ecolor="0.7", zorder=6)
        lineprop  = dict(color="0.7", zorder=1, lw=0.5)

        if saltmodel is None:
            autoscale_salt = False

        if not np.isnan(t0):
            if phase_range is not None: # removes NaN
                timerange = [t0+phase_range[0], t0+phase_range[1]]
            else:
                timerange = None

            modeltime = t0 + np.linspace(-15,50,100)
        else:
            timerange = None
            if saltmodel is not None:
                warnings.warn("t0 in saltdata is NaN, cannot show the model")
            if as_phase:
                warnings.warn("t0 in saltdata is NaN, as_phase not available")
                as_phase = False

            saltmodel = None
            autoscale_salt = False

        if not rm_flags:
            prop = {"flagout":None}
        else:
            prop = {}

        if bands is None or bands in ["*", "all"]:
            bands = np.unique(lcdata["filter"])
        else:
            bands = np.atleast_1d(bands)


        max_saltlc = 0
        min_saltlc = 100
        # Loop over bands
        for band_ in bands:
            if band_ not in ZTFCOLOR:
                warnings.warn(f"WARNING: Unknown instrument: {band_} | magnitude not shown")
                continue

            flagband   = (lcdata["filter"]==band_)

            bdata = lcdata[flagband]
#            flag_good_ = flag_good[flagband]

            # IN FLUX
            if not inmag:
                # - Data
                if as_phase:
                    datatime = bdata["mjd"].astype("float") - t0
                else:
                    datatime = Time(bdata["mjd"].astype("float"), format="mjd").datetime

                y, dy = bdata["flux"], bdata["error"]
                # - Salt
                if saltmodel is not None:
                    saltdata = saltmodel.bandflux(band_, modeltime, zp=zp, zpsys="ab") \
                      if saltmodel is not None else None
                else:
                    saltdata = None

            # IN MAG
            else:
                flag_det = (lcdata["mag"]<99)
                # - Data
                bdata = bdata[flag_det]
                #flag_good_ = flag_good_[flag_det]
                if as_phase:
                    datatime = bdata["mjd"].astype("float") - t0
                else:
                    datatime = Time(bdata["mjd"], format="mjd").datetime

                y, dy = bdata["mag"], bdata["mag_err"]
                # - Salt
                if saltmodel is not None:
                    saltdata = saltmodel.bandmag(band_, "ab",modeltime) if saltmodel is not None else None
                else:
                    saltdata = None

            # -> good
            ax.errorbar(datatime,#[flag_good_],
                            y,#[flag_good_],
                            yerr=dy,#[flag_good_],
                            label=band_,
                            **{**base_prop, **ZTFCOLOR[band_],**kwargs}
                            )
            # -> bad
            ax.errorbar(datatime,#[~flag_good_],
                            y,#[~flag_good_],
                            yerr=dy,#[~flag_good_],
                            label=band_,
                            **{**bad_prop, **BAD_ZTFCOLOR[band_],**kwargs}
                            )

            if saltdata is not None:
                if as_phase:
                    modeltime_ = modeltime - t0
                else:
                    modeltime_ = Time(modeltime, format="mjd").datetime

                ax.plot(modeltime_,
                        saltdata,
                        color=ZTFCOLOR[band_]["mfc"], zorder=5)

                max_saltlc = np.max([max_saltlc, np.max(saltdata)])
                min_saltlc = np.min([min_saltlc, np.min(saltdata)])

        if inmag:
            ax.invert_yaxis()
            for band_ in bands:
                bdata = lcdata[(lcdata["filter"]==band_) & (lcdata["mag"]>=99)]
                if as_phase:
                    datatime = Time(bdata["mjd"], format="mjd").datetime
                else:
                    datatime = bdata["mjd"].astype("float") - t0

                y = bdata["mag_lim"]
                ax.errorbar(datatime, y,
                                 yerr=ulength, lolims=True, alpha=ualpha,
                                 color=ZTFCOLOR[band_]["mfc"],
                                 ls="None",  label="_no_legend_")

        if formattime and not as_phase:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        lunit = "flux" if not inmag else "mag"
        ax.set_ylabel(f"{lunit} [zp={zp}]" if zp is not None else f"{lunit} []")
        if zeroline:
            ax.axhline(0 if not inmag else 22, **{**dict(color="0.7",ls="--",lw=1, zorder=1),**zprop} )

        if not inmag:
            max_data = np.percentile(lcdata["flux"], 99.)
            mean_error = np.nanmean(lcdata["error"])
            ax.set_ylim(-2*mean_error, max_data*1.15)
            if clear_yticks:
                ax.axes.yaxis.set_ticklabels([])

        if autoscale_salt:
            if timerange is not None:
                if as_phase:
                    ax.set_xlim(*(np.asarray(timerange)-t0))
                else:
                    ax.set_xlim(*Time(timerange,format="mjd").datetime)

            if not inmag:
                ax.set_ylim(bottom=-max_saltlc*0.25)
                ax.set_ylim(top=max_saltlc*1.25)
            else:
                if np.isinf(min_saltlc) or np.isnan(min_saltlc):
                    ax.set_ylim(23, 14)
                else:
                    ax.set_ylim(top=min_saltlc*0.95)

        return fig

    # ============== #
    # Properties     #
    # ============== #
    @property
    def data(self):
        return self._data

    @property
    def saltdata(self):
        return self._saltdata
