import numpy as np
from numpy import *
import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
from matplotlib.pyplot import *

x = linspace(-1, 1, 1000)

def f1f(x):
	# f = exp(2*x)
	# df = 2*exp(2*x)
	# d2f = 4*exp(2*x)
	# mf = exp(2)
	# f /= mf
	# df /= mf
	# d2f /= mf
	f = log(1+exp(2*x))# + 0.25*x + 0.25
	df = 2*exp(2*x) / (1+exp(2*x))# + 0.25
	ss = df/2.
	#d2f = -4*df*(1-df)
	d2f = 4*ss*(1-ss)
	mf = log(1+exp(2))/sqrt(2)

	ax = abs(x)
	sx = sign(x)
	f = x**4 - x**3 + x**2 + x + 0.5
	df = 4*x**3 - 3*x**2 + 2*x + 1
	d2f = 12*x**2 - 6*x + 2

	mf = 1.5
	f /= mf
	df /= mf
	d2f /= mf


	return f, df, d2f
def f2f(x):
	# f = exp(3 - 3*x)-1
	# df = -3*exp(3 - 3*x)
	# d2f = 9*exp(3 - 3*x)
	# f = cos(x*pi)**2
	# df = -2*pi*cos(x*pi)*sin(x*pi)
	# d2f = 2*pi**2 * (sin(x*pi)**2 - cos(x*pi)**2) 
	f = abs(x)**3 - 0.5*x + 0.25
	df = 3*x**2*sign(x) - 0.5
	d2f = 6*abs(x)
	return f, df, d2f
def gf(x, xs, ff, last_x = None):
	f, df, d2f = ff(xs)
	if not last_x == None:
		_, dfl, _ = ff(last_x)
		d2f = (df - dfl)/(xs - last_x)
	g = f + (x-xs)*df + 0.5*(x-xs)**2*d2f
	return g

f1,_,_ = f1f(x)
f2,_,_ = f2f(x)

x1 = 0.8
g1 = gf(x, x1, f1f)
x2 = 0.6
g2 = gf(x, x2, f2f)
G = g1+g2

figsize=(4.,3.4)
figure(figsize=figsize)
plot(x, f1, label='$f_1(x)$', color='b', ls = '--', linewidth=3)
plot(x, f2, label='$f_2(x)$', color='b', ls = ':', linewidth=3)
plot(x, f1 + f2, label='$F(x)$', color='b', ls = '-', linewidth=3)
plot(x, g1, label='$g^{t-1}_1(x)$', color='r', ls = '--')
plot(x, g2, label='$g^{t-1}_2(x)$', color='r', ls = ':')
plot(x, g1 + g2, label='$G^{t-1}(x)$', color='r', ls = '-')
axis([-0.75,1,0,2.5])
xticks([x2,x1], ['$x^{t-1}$', '$x^{t-2}$'])
yticks([], [])
grid()
legend(ncol=2, loc=2)
plot(x1, g1[(x-x1)**2==np.min((x-x1)**2)], marker='o', color='g', linewidth=5)
plot(x2, g2[(x-x2)**2==np.min((x-x2)**2)], marker='o', color='g', linewidth=5)
tight_layout()
subplots_adjust(left=0.1)
savefig('figure_cartoon_pane_A.pdf')

# find the new minimum
idx = nonzero(G == np.min(G))[0][0]
x1s = x[idx]
Gs = G[idx]
g1s = gf(x, x1s, f1f, last_x = x1)


figure(figsize=figsize)
plot(x, f1, label='$f_1(x)$', color='b', ls = '--', linewidth=3)
plot(x, f2, label='$f_2(x)$', color='b', ls = ':', linewidth=3)
plot(x, f1 + f2, label='$F(x)$', color='b', ls = '-', linewidth=3)
plot(x, g1, label='$g^{t-1}_1(x)$', color='r', ls = '--')
plot(x, g2, label='$g^{t-1}_2(x)$', color='r', ls = ':')
plot(x, g1 + g2, label='$G^{t-1}(x)$', color='r', ls = '-')
axis([-0.75,1,0,2.5])
xticks([x2,x1,x1s], ['$x^{t-1}$', '$x^{t-2}$', '$x^{t}$'])
yticks([], [])
grid()
#legend(bbox_to_anchor=(0.675, 1.125), ncol=2)
legend(ncol=2, loc=2)
plot(x1, g1[(x-x1)**2==np.min((x-x1)**2)], marker='o', color='g', linewidth=5)
plot(x2, g2[(x-x2)**2==np.min((x-x2)**2)], marker='o', color='g', linewidth=5)
tight_layout()
subplots_adjust(left=0.1)
yticks([Gs,], ['min $G^{t-1}(x)$',], rotation=90)
savefig('figure_cartoon_pane_B.pdf')

figure(figsize=figsize)
plot(x, f1, label='$f_1(x)$', color='b', ls = '--', linewidth=3)
plot(x, f2, label='$f_2(x)$', color='b', ls = ':', linewidth=3)
plot(x, f1 + f2, label='$F(x)$', color='b', ls = '-', linewidth=3)
plot(x, g1s, label='$g^{t}_1(x)$', color='r', ls = '--')
plot(x, g2, label='$g^{t}_2(x)$', color='r', ls = ':')
plot(x, g1s + g2, label='$G^{t}(x)$', color='r', ls = '-')
axis([-0.75,1,0,2.5])
xticks([x2,x1,x1s], ['$x^{t-1}$', '$x^{t-2}$', '$x^{t}$'])
yticks([], [])
grid()
legend(ncol=2, loc=2)
plot(x1, g1[(x-x1)**2==np.min((x-x1)**2)], marker='o', color='g', linewidth=5)
plot(x2, g2[(x-x2)**2==np.min((x-x2)**2)], marker='o', color='g', linewidth=5)
plot(x1s, g1s[(x-x1s)**2==np.min((x-x1s)**2)], marker='o', color='g', linewidth=5)
tight_layout()
subplots_adjust(left=0.1)
savefig('figure_cartoon_pane_C.pdf')

#show()