# Alberto Ruiz Biestro -- A01707550
#
# N_SPHERE volume
#
# Para más información revisar: https://en.wikipedia.org/wiki/N-sphere
# ToDo: initialize arrays
# Last revision: 7/03/2022


########################### IMPORT ###################################

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import functools

plt.rcParams['figure.dpi'] = 120
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams.update({
  'font.size': 15,
  'text.usetex': True,
  'text.latex.preamble': r'\usepackage{amsfonts}'
  })

######################### FUNCTIONS ##################################

def get_samples(n, N):
  """
  | Obtiene las 'muestras', mediante la distribución uniforme, acotada
  | por un hípercubo de dimensión 'n' centrado en el origen.
  |
  | Retorna un vector (Tensor, más bien) de las 
  | muestras
  | -----------
  | Parámetros:
  | -----------
  |   n: dimensión
  |   N: número de puntos
  """
  samp = [None] * n
  for ii in range(n):
      # unit sphere, too lazy to change it
    x2 = 1
    x1 = -1
    # hacemos N*n para "escalar" la distribución relativamente a
    # dada dimensión
    samp[ii] = (x2 - x1) * np.random.rand(N*n) + x1
  return samp

def inside_ball(samp):
  """
  | Calcula las posiciones de los vectores/tensores en donde está el círculo (y dónde no).

  | Retorna estos tensores al igual que dichas posiciones
  | -----------
  | Parámetros:
  | -----------
  |   samp: muestras (obtenidas mediante get_samples())
  """
  # contamos cuantos puntos están dentro
  #print(len(samp))
  radius = 0
  for ii in range(len(samp)): radius += samp[ii]**2
  
  # posiciones donde el radio sea menor a 1 (|x| < 1)
  pos = (radius**0.5 <= 1)
  #pos = ((samp[0]**2 + samp[1]**2 + samp[2]**2)**0.5 <= 1) # para 3 d

  # vector 
  # esto hace que se tarde mucho, pero no cuento con el tiempo de
  # inicializar los arreglos (matrices)
  x = [None] * len(samp)
 # x = np.zeros(len(samp))
  #x = np.zeros(len(samp))
  x_ = [None] * len(samp)
  
  for ii in range(len(samp)):
    x[ii] = samp[ii] * pos # obtenemos lugares en donde sí se cumple
    x_[ii] = samp[ii] * ~pos

  # ignorar lo de abajo (rough work)
  #y = samp[1] * pos # *1 o *0, entonces sólo los True vals son guardados
  #z = samp[2] * pos
  # puntos que no cumplen:
  #x_ = samp[0] * ~pos# obtenemos lugares en donde sí se cumple
  #y_ = samp[1] * ~pos# *1 o *0, entonces sólo los True vals son guardados
  #z_ = samp[2] * ~pos
  return x, x_, pos

def get_vol(n_, N_):
  
  samples_ = get_samples(n_, N_)
  ball_ = inside_ball(samples_)
  n_in = ball_[2]
  N_IN = np.sum(n_in)
  A_TOT = 2**n_
  N_TOT = N_
  
  # dividimos entre n_ (dimensión) para contrarrestar
  # el ajuste hecho previamente
  #
  # ver la función get_samples()
  return A_TOT * N_IN / N_TOT / n_
    
########################## MAIN ###################################

if __name__ == "__main__":
  n = int(input("Dimension (n)?: "))
  if n == 3: print("\nWARNING: Plotting many samples in 3-d space can take a while\n") 
  N = int(input("Number of samples/points (N)?: "))

  samples = get_samples(n, N)

  ball = inside_ball(samples)

  #print(ball[0][0])

  volume = get_vol(n, N)

  # plotting
  # if dimension = 2
  if n == 2:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(ball[1][0], ball[1][1], alpha=0.5,label=r'$||x||>1$')
    ax.scatter(ball[0][0], ball[0][1], alpha=0.5,label=r'$||x||\leq 1$')
    ax.legend(loc='best' )
    ax.set_aspect('equal')
    plt.title(r"2-D sphere, vol $\approx$ "+str(volume) + " (N=" + str(N*n) + ")")
    plt.savefig("2_d_vol.png")
    plt.show()

  # if dimension = 3
  if n == 3:
    fig = plt.figure()

    # hay un punto en el centro que no sé qué hace ahí jaja
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ball[1][0], ball[1][1], ball[1][2], alpha=0.3,label=r'$||x||>1$')
    ax.scatter(ball[0][0], ball[0][1], ball[0][2], alpha=0.5,label=r'$||x||\leq 1$')
    #ax.set_xlim([-1.5, 1.5])
    #ax.set_ylim([-1.5, 1.5])
    ax.legend()
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.title(r"3-D sphere, vol $\approx$ "+str(volume)+ " (N=" + str(N*n) + ")")

    plt.savefig("3_d_vol.png")
    plt.show()


  print(f"\nApproximated Volume of Hypersphere = {volume}\n")

  ###################
  # for various dimensions
  n_ball_volume = lambda n,R: np.pi**(n/2) * R**n / sp.gamma((n/2) + 1)

  # max dimensions
  n_max = 30

  n_vec = np.arange(1,n_max,1)

  vol = [None] * len(n_vec)
  
  N_new = 10000

  vol= [get_vol(n_vec[ii], N_new) for ii in range(len(n_vec))]
      
  analytic = n_ball_volume(n_vec,1)

  fig, ax = plt.subplots(1,1)
  ax.plot(n_vec, analytic, alpha=.8, label="Analytic")
  ax.plot(n_vec, vol, "-s",fillstyle="none", alpha=.65, label=f"Montecarlo method,\n N={N_new}")
  ax.legend()
  ax.set_xlabel(r"$n$")
  ax.set_ylabel(r"$V_n$")
  # ax.set_title(r"Volume of $n-sphere$ using N=" + str(N_new ) )

  # definición (de wikipedia)
  ax.set_title(r"Volume $(V_n)$ of $S^n = \{x\in R^{n+1} \ \colon \  ||x||\leq 1\}$")
  ax.grid(alpha=0.5)
  plt.savefig("Vol_n.png")
  plt.show()

  # average plot

  cond = input("\nRun extremely long average Montecarlo section? (Y/N) ").lower()

  if cond == "y":
    N_new = 5000
    fig1, ax1 = plt.subplots(1,1)
    ax1.plot(n_vec, analytic, alpha = .8, label='Analytic')
    ax1.legend()
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(r"$V_n$")
    ax1.set_title(r"Volume $(V_n)$ of $S^n = \{x\in R^{n+1} \ \colon \  ||x||\leq 1\}$")

    for jj in range(50):
      vol = [get_vol(n_vec[ii], N_new) for ii in range(len(n_vec))]
      #ax1.plot(n_vec, vol, 'oc', fillstyle='none', alpha=.65)
      ax1.plot(n_vec, vol, '#bfbbd9',marker='o',linestyle='None',  alpha=.55)

      
    ax1.grid(alpha=0.5)
    plt.savefig("Average_Vol_n.png")
    plt.show()

