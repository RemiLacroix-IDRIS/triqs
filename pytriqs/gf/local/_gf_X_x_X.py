import numpy as np
from pytriqs.gf.local.multivar import *

def plot(self, opt_dict):
    r"""
    Plot protocol for GfBrillouinZone objects.
    """

    plot_type = opt_dict.pop('type','contourf')
    method = opt_dict.pop('method', 'nearest')
    comp = opt_dict.pop('mode', 'R')
    component=  lambda x : x.real if comp=="R" else x.imag
    if type(self)==GfImFreq_x_ImFreqTv3:
     X_label = r"i\omega"
     Y_label = r"i\Omega"
     mp_comp = lambda x : x.imag
    elif type(self)==GfImTime_x_ImTimeTv3:
     X_label = r"\omega"
     Y_label = r"\Omega"
     mp_comp = lambda x : x.real
    else:
     X_label = "X"
     Y_label = "Y"
     mp_comp = lambda x : x.real

    if plot_type=="contourf":
     x = np.array([mp_comp(m) for m in self.mesh.components[0]])
     y = np.array([mp_comp(m) for m in self.mesh.components[1]])
     z = np.transpose(self.data[:,:,0,0,0])
     zmin=np.amin(z.real)+np.amin(z.imag)*1j
     zmax=np.amax(z.real)+np.amax(z.imag)*1j

     default_dict = {'xdata': x, 
                     'ydata': y, 
                     'label': r'$G_\mathbf{k}$', 
                     'xlabel': r'$%s$'%X_label, 
                     'ylabel': r'$%s$'%Y_label, 
                     'zdata' : component(z),
                     'levels':np.linspace(component(zmin),component(zmax),50), 
                     'plot_function': plot_type,
                     'title': r'$\mathrm{%s}G(%s,%s)$'%('Re' if comp=='R' else 'Im',X_label, Y_label), 
                    }
    else: raise Exception("Unknown plot type %s. Should be 'contourf' (default).'"%plot_type)

    default_dict.update(opt_dict)

    return [default_dict]
