import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pdb
import os
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # pour 3D

# Parameters
# TODO adapt to what you need (folder path executable input filename)
executable = 'Exercice5_2025_student'  # Name of the executable (NB: .exe extension is required on Windows)
repertoire = r"/Users/Sayu/Desktop/Wave"
os.chdir(repertoire)


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configuration.in")

input_filename = 'configuration.in'  # Name of the input file

def lire_configuration():
    config_path = os.path.join(os.path.dirname(__file__), "configuration.in")
    configuration = {}
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier {config_path} n'existe pas.")
    
    with open(config_path, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne = ligne.strip()
            if ligne and "=" in ligne and not ligne.startswith("#"):
                cle, valeur = ligne.split("=", 1)
                configuration[cle.strip()] = valeur.strip()
    
    return configuration

def ecrire_configuration(nouvelles_valeurs):
   
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Le fichier {CONFIG_FILE} n'existe pas.")

    lignes_modifiees = []
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne_strippée = ligne.strip()
            if ligne_strippée and "=" in ligne_strippée and not ligne_strippée.startswith("#"):
                cle, _ = ligne_strippée.split("=", 1)
                cle = cle.strip()
                if cle in nouvelles_valeurs:
                    ligne = f"{cle} = {nouvelles_valeurs[cle]}\n"
            lignes_modifiees.append(ligne)

    with open(CONFIG_FILE, "w", encoding="utf-8") as fichier:
        fichier.writelines(lignes_modifiees)

# Paramètres physiques actuels (issus du fichier configuration.in)
cb_gauche = "libre"
cb_droite = "fixe"
v_uniform = True
A = 0.1
f_hat = 1.0
x1 = 3.0
x2 = 8.0
tfin = 1.0
equation_type = "B"
nx = 64
n_init = 3
initialization = "mode"
initial_state = "static"
CFL = 1.0
nsteps = 40
impose_nsteps = True
output = "test.txt"
n_stride = 1
ecrire_f = 1
hL = 8000.0
hR = 20.0
h00 = 4.0
xa = 450e3
xb = 950e3
L = 15.0
om = 0.1
fac = "true"

valeurs = lire_configuration()

def actualise_valeur():
    global cb_gauche, cb_droite, v_uniform, A, f_hat, x1, x2, tfin, equation_type
    global nx, n_init, initialization, initial_state, CFL, nsteps, impose_nsteps
    global output, n_stride, ecrire_f, hL, hR, h00, xa, xb, L, om, fac

    cb_gauche = valeurs.get("cb_gauche")
    cb_droite = valeurs.get("cb_droite")
    v_uniform = valeurs.get("v_uniform") == "true"
    A = float(valeurs.get("A"))
    f_hat = float(valeurs.get("f_hat"))
    x1 = float(valeurs.get("x1"))
    x2 = float(valeurs.get("x2"))
    tfin = float(valeurs.get("tfin"))
    equation_type = valeurs.get("equation_type")
    nx = int(valeurs.get("nx"))
    n_init = int(valeurs.get("n_init"))
    initialization = valeurs.get("initialization")
    initial_state = valeurs.get("initial_state")
    CFL = float(valeurs.get("CFL"))
    nsteps = int(valeurs.get("nsteps"))
    impose_nsteps = valeurs.get("impose_nsteps") == "true"
    output = valeurs.get("output")
    n_stride = int(valeurs.get("n_stride"))
    ecrire_f = int(valeurs.get("ecrire_f"))
    hL = float(valeurs.get("hL"))
    hR = float(valeurs.get("hR"))
    h00 = float(valeurs.get("h00"))
    xa = float(valeurs.get("xa"))
    xb = float(valeurs.get("xb"))
    L = float(valeurs.get("L"))
    om = float(valeurs.get("om"))
    fac = valeurs.get("fac") == "true"

def ecrire_valeur(nom, valeur):
    global valeurs
    valeurs[nom] = str(valeur)
    ecrire_configuration(valeurs)
    actualise_valeur()

def lancer_simulation(param_valeur, output_file):
    ecrire_configuration({"output": output_file})
    cmd = f"./{executable} {input_filename} output={output_file}"
    subprocess.run(cmd, shell=True)

def ecrire_valeur(nom,valeur):
    global valeurs
    valeurs[nom] = valeur
    ecrire_configuration(valeurs)
    actualise_valeur()

def lancer_simulation(theta0, output_file):
    ecrire_configuration({"theta0": theta0})
    cmd = f"./{executable} {input_filename} output={output_file}"
    subprocess.run(cmd, shell=True)

outputs = []  # Liste pour stocker les fichiers de sortie
errors = []  # Liste pour stocker les erreurs
values = []


######################################################## Question 5.3 (a) ########################################################
"""

ecrire_valeur("fac","false")
ecrire_valeur("n_stride",1)
ecrire_valeur("equation_type","A")
ecrire_valeur("L",15.0)
ecrire_valeur("h00",4.0)
ecrire_valeur("cb_gauche","libre")
ecrire_valeur("cb_droite","fixe")
ecrire_valeur("f_hat",1.0)
ecrire_valeur("x1",3.0)
ecrire_valeur("x2",8.0)
ecrire_valeur("v_uniform","true")
ecrire_valeur("nx",64)
ecrire_valeur("nsteps",400)
ecrire_valeur("impose_nsteps","true")
ecrire_valeur("tfin",10.0)
ecrire_valeur("initialization","autre")

# Vague initialement statique
ecrire_valeur("initial_state","static")

output_file = f"Question_1_static.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Question_1_static output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_static = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_static.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (statique)")
plt.grid(True)
plt.tight_layout()

fig_static, ax_static = plt.subplots()
line_static, = ax_static.plot([], [], lw=2)
ax_static.set_xlim(x[0], x[-1])
ax_static.set_ylim(np.min(f_static), np.max(f_static) * 1.1)
ax_static.set_xlabel("x (m)")
ax_static.set_ylabel("f(x, t)")
ax_static.set_title("Animation vague statique")

def init_static():
    line_static.set_data([], [])
    return line_static,

def update_static(frame):
    line_static.set_data(x, f_static[frame, :])
    ax_static.set_title(f"Statique, t = {frame*dt:.2f} s")
    return line_static,

ani_static = animation.FuncAnimation(fig_static, update_static, frames=nt, init_func=init_static,
                                     blit=True, interval=30)

plt.tight_layout()

# Vague initialement orientée vers la gauche
ecrire_valeur("initial_state","left")

output_file = f"Question_1_left.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Question_1_left output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_left = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_left.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (gauche)")
plt.grid(True)
plt.tight_layout()

fig_left, ax_left = plt.subplots()
line_left, = ax_left.plot([], [], lw=2)
ax_left.set_xlim(x[0], x[-1])
ax_left.set_ylim(np.min(f_left), np.max(f_left) * 1.1)
ax_left.set_xlabel("x (m)")
ax_left.set_ylabel("f(x, t)")
ax_left.set_title("Animation vague vers la gauche")

def init_left():
    line_left.set_data([], [])
    return line_left,

def update_left(frame):
    line_left.set_data(x, f_left[frame, :])
    ax_left.set_title(f"Gauche, t = {frame*dt:.2f} s")
    return line_left,

ani_left = animation.FuncAnimation(fig_left, update_left, frames=nt, init_func=init_left,
                                   blit=True, interval=30)
plt.tight_layout()


# Vague initialement orientée vers la droite
ecrire_valeur("initial_state","right")

output_file = f"Question_1_right.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Question_1_right output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_right = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_right.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (droite)")
plt.grid(True)
plt.tight_layout()

fig_right, ax_right = plt.subplots()
line_right, = ax_right.plot([], [], lw=2)
ax_right.set_xlim(x[0], x[-1])
ax_right.set_ylim(np.min(f_right), np.max(f_right) * 1.1)
ax_right.set_xlabel("x (m)")
ax_right.set_ylabel("f(x, t)")
ax_right.set_title("Animation vague vers la droite")

def init_right():
    line_right.set_data([], [])
    return line_right,

def update_right(frame):
    line_right.set_data(x, f_right[frame, :])
    ax_right.set_title(f"Droite, t = {frame*dt:.2f} s")
    return line_right,

ani_right = animation.FuncAnimation(fig_right, update_right, frames=nt, init_func=init_right,
                                    blit=True, interval=30)
plt.tight_layout()


# Afficher toutes les figures en même temps
plt.show()
"""
######################################################## Question 5.3 (b) ########################################################
"""
ecrire_valeur("fac","false")
ecrire_valeur("n_stride",1)
ecrire_valeur("L",15.0)
ecrire_valeur("h00",4.0)
ecrire_valeur("cb_gauche","fixe")
ecrire_valeur("cb_droite","libre")
ecrire_valeur("f_hat",1.0)
ecrire_valeur("x1",3.0)
ecrire_valeur("x2",8.0)
ecrire_valeur("v_uniform","true")
ecrire_valeur("impose_nsteps","false")
ecrire_valeur("nx",64)
ecrire_valeur("nsteps",4000)
ecrire_valeur("CFL",1.001)
ecrire_valeur("equation_type","A")
ecrire_valeur("tfin",10.0)
ecrire_valeur("initialization","autre")

# Cas traité Vague initialement orientée vers left

ecrire_valeur("initial_state","left")

output_file = f"Question_2_left.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Question_2_left output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_q2 = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_q2.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (gauche)")
plt.grid(True)
plt.tight_layout()

fig_q2, ax_q2 = plt.subplots()
line_q2, = ax_q2.plot([], [], lw=2)
ax_q2.set_xlim(x[0], x[-1])
#ax_q2.set_ylim(np.min(f_q2), np.max(f_q2) * 1.1)
ax_q2.set_xlabel("x (m)")
ax_q2.set_ylabel("f(x, t)")
ax_q2.set_title(f"f(x, t) évaluée à plusieurs t pour CFL = {CFL}")


#def init_q2():
#    line_q2.set_data([], [])
#    return line_q2,

#def update_q2(frame):
#    line_q2.set_data(x, f_q2[frame, :])
#    ax_q2.set_title(f"Droite, t = {frame*dt:.2f} s")
#    return line_q2,

#ani_q2 = animation.FuncAnimation(fig_q2, update_q2, frames=nt, init_func=init_q2,
#                                   blit=True, interval=30)


iss = np.array([0,100,120])
for i in iss:
    plt.plot(x,f_q2[i], label = f"f(x,t={i*dt})")
plt.legend()

plt.tight_layout()
plt.show()
"""

######################################################## Question 5.3 (c) ########################################################

"""# Exemple pour la solution numérique et anlytique
g  = 9.81

ecrire_valeur("fac","false")
ecrire_valeur("n_stride",1)
ecrire_valeur("L",15.0)
ecrire_valeur("h00",4.0)
ecrire_valeur("cb_gauche","libre")
ecrire_valeur("cb_droite","fixe")
ecrire_valeur("f_hat",1.0)
ecrire_valeur("x1",3.0)
ecrire_valeur("x2",8.0)
ecrire_valeur("v_uniform","true")
ecrire_valeur("impose_nsteps","false")
ecrire_valeur("nx",64)
ecrire_valeur("nsteps",200)
ecrire_valeur("CFL",1)
ecrire_valeur("equation_type","A")
ecrire_valeur("tfin",10.0)
ecrire_valeur("initialization","mode")
ecrire_valeur("n_init",3)


# Cas traité Vague initialement orientée vers left

ecrire_valeur("initial_state","static")

output_file = f"Question_3_left.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Question_3_left output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_q3 = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_q3.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]


fig_q3, ax_q3 = plt.subplots()
line_q3, = ax_q3.plot([], [], lw=2)
ax_q3.set_xlim(x[0], x[-1])
#ax_q3.set_ylim(np.min(f_q2), np.max(f_q2) * 1.1)
ax_q3.set_xlabel("x (m)")
ax_q3.set_ylabel("f(x, t)")
ax_q3.set_title(f"f(x, t) évaluée à plusieurs t pour CFL = {CFL}, nx = {nx} et nsteps = {nsteps}")


#def init_q2():
#    line_q2.set_data([], [])
#    return line_q2,

#def update_q2(frame):
#    line_q2.set_data(x, f_q2[frame, :])
#    ax_q2.set_title(f"Droite, t = {frame*dt:.2f} s")
#    return line_q2,

#ani_q2 = animation.FuncAnimation(fig_q2, update_q2, frames=nt, init_func=init_q2,
#                                   blit=True, interval=30)


t0 = np.pi/(np.sqrt(g*h00)*((n_init+0.5)*np.pi/(2*L)))
print(f"t0= {t0}")
plt.plot(x,f_q3[int(np.floor(t0/dt))], label = "$f_{num}(x,t = T ="+f"{t0})$")
plt.plot(x,f_hat*np.cos((2*n_init+1)*x*np.pi/(2*L)), label = "$f_{ana}(x,T)$")

plt.legend()

plt.tight_layout()

plt.show()


tab = [90,120,200,240]
outputs = []
errors = []
nxs = []

for el in tab:

    ecrire_valeur("nx", el)

    output_file = f"nx={el}.txt"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} nx={el} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Simulation terminée.')

    f_q3b = np.loadtxt("f_" + output_file)[:, 1:]
    x = np.loadtxt("x_" + output_file) 
    E = np.loadtxt("en_" + output_file)[:, 1:]
    
    nt, nx = f_q3.shape
    dx = x[1] - x[0]
    time = np.loadtxt("f_"+output_file)[:,0]
    dt = time[1]-time[0]

    t0 = np.pi/(np.sqrt(g*h00)*((n_init+0.5)*np.pi/L))
    y_num = f_q3b[int(np.floor(t0/dt))]
    y_ana = f_hat*np.cos((2*n_init+1)*x*np.pi/(2*L))
    errors.append(dx*np.sum(np.abs(y_num - y_ana)))
    nxs.append(el)
print(errors)
n= np.array(nxs)
plt.figure()
plt.loglog(n,np.abs(19.1347-np.array(errors)), marker='v', markersize=3, color="black", linestyle='-')
plt.ylabel("erreur")
plt.xlabel("$n$")
plt.grid(True, linestyle="--", alpha=0.3)
plt.title(f"Convergence de l'erreur par rapport à nx et nsteps pour $\\beta$ = {CFL}")
plt.show()

"""
######################################################## Question 5.3 (d) ########################################################
"""
# Exemple pour la solution numérique et anlytique
g  = 9.81

ecrire_valeur("fac","false")

ecrire_valeur("n_stride",1)
ecrire_valeur("L",15.0)
ecrire_valeur("h00",4.0)
ecrire_valeur("cb_gauche","libre")
ecrire_valeur("cb_droite","excitation")
ecrire_valeur("x1",3.0)
ecrire_valeur("x2",8.0)
ecrire_valeur("v_uniform","true")
ecrire_valeur("impose_nsteps","false")
ecrire_valeur("nx",44)
ecrire_valeur("nsteps",64)
ecrire_valeur("CFL",0.9)
ecrire_valeur("equation_type","A")
ecrire_valeur("tfin",500.0)
ecrire_valeur("initialization","mode")
ecrire_valeur("n_init",3)
ecrire_valeur("A",0.1)
ecrire_valeur("initial_state","static")
ecrire_valeur("f_hat",0)

# Différentes valeurs de la fréquence seront traitées.

ecrire_valeur("om",0.1)

OMS = np.linspace(0,2.4,30)

E_hat = []

for el in OMS:

    ecrire_valeur("om", el)

    output_file = f"om={el}.txt"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} om={el} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Simulation terminée.')

    E = np.loadtxt("en_" + output_file)[:, 1:]
    E_hat.append(np.max(E))

OMS= np.array(OMS)

plt.figure()
plt.plot(OMS, E_hat, markersize=3, color="black", linestyle='-')
for n in range(2):
    om_n = np.sqrt(g*h00)*((n+0.5)*np.pi/L)
    print(f"n = {n} et om_n = {om_n}")
    plt.axvline(om_n, color="red", linestyle='--')
plt.ylabel("max(E)")
plt.xlabel("$\\omega$")
plt.grid(True, linestyle="--", alpha=0.3)
plt.title(f"Étude du maximum d'énergie atteint pour différentes fréquences $\\omega$ d'excitation")

plt.show()

g = 9.81
for n in range(1,20):
    om_n = np.sqrt(g*h00)*(n*np.pi/L)
    print(f"n = {n} et om_n = {om_n}")

"""
##############################################################################################################################
######################################################## Question 5.4 (a) ####################################################
"""
# Exemple pour la solution numérique et anlytique
g  = 9.81

ecrire_valeur("fac","false")
ecrire_valeur("L",1000000)
ecrire_valeur("hR",20.0)
ecrire_valeur("hL",8000.0)
ecrire_valeur("xa",450000)
ecrire_valeur("xb",950000)
ecrire_valeur("cb_gauche","sortie")
ecrire_valeur("cb_droite","sortie")
ecrire_valeur("x1",50000)
ecrire_valeur("x2",250000)
ecrire_valeur("v_uniform","false")
ecrire_valeur("impose_nsteps","false")
ecrire_valeur("nx",1000)
ecrire_valeur("nsteps",64)
ecrire_valeur("CFL",1)
ecrire_valeur("equation_type","B")
ecrire_valeur("tfin",12000)
ecrire_valeur("initialization","autre")
ecrire_valeur("initial_state","right")
ecrire_valeur("f_hat",1.0)
ecrire_valeur("n_stride",10)


x = np.linspace(xa,xb,1000)
t = np.linspace(0,xa,1000)
k = np.linspace(xb,L,1000)
plt.figure()
plt.plot(t,hL*np.ones(len(t)))
plt.plot(x,0.5*(hL+hR)+0.5*(hL-hR)*np.cos(np.pi*(x-xa)/(xb-xa)))
plt.plot(k,hR*np.ones(len(k)))
#plt.show()


output_file = f"Tsunami.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Tsunami output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_tsu = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
v = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_tsu.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (statique)")
plt.grid(True)
plt.tight_layout()

fig_tsu, ax_tsu = plt.subplots()
line_tsu, = ax_tsu.plot([], [], lw=2)
ax_tsu.set_xlim(x[0], x[-1])
ax_tsu.set_ylim(np.min(f_tsu), np.max(f_tsu) * 1.1)
ax_tsu.set_xlabel("x (m)")
ax_tsu.set_ylabel("f(x, t)")
ax_tsu.set_title("Animation vague statique")

def init_tsu():
    line_tsu.set_data([], [])
    return line_tsu,

def update_tsu(frame):
    line_tsu.set_data(x, f_tsu[frame, :])
    ax_tsu.set_title(f"Statique, t = {frame*dt:.2f} s")
    return line_tsu,

ani_static = animation.FuncAnimation(fig_tsu, update_tsu, frames=nt, init_func=init_tsu,
                                     blit=True, interval=30)


x1 = np.linspace(0, xa, 300)
x2 = np.linspace(xa, xb, 300)
x3 = np.linspace(xb, L, 300)

h = np.concatenate([
    hL * np.ones_like(x1),
    0.5 * (hL + hR) + 0.5 * (hL - hR) * np.cos(np.pi * (x2 - xa) / (xb - xa)),
    hR * np.ones_like(x3)
])
x_full = np.concatenate([x1, x2, x3])
ax_tsu.plot(x_full, -h, color="blue", linestyle="--", label="Profondeur")

plt.tight_layout()


######################################################## Question 5.4 (b) ####################################################


# Trouver les valeurs maximales par ligne
f_max = np.max(f_tsu[0:int(np.floor(9140/dt))], axis=1)

# Trouver les indices des valeurs maximales par ligne
f_max_indices = np.argmax(f_tsu[0:int(np.floor(9140/dt))], axis=1)

x_max_values = x[f_max_indices]

plt.figure()
plt.plot(x_max_values,f_max)
plt.xlabel("Position x")
plt.ylabel("Hauteur de la vague y")
plt.title("Hauteur de la vague en fonction de la position")

######################################################## Question 5.4 (c) ####################################################
# Création du vecteur des temps correspondant à chaque position max
t = np.arange(0, len(x_max_values)) * dt

# Calcul des vitesses de propagation entre deux instants successifs
v_propag = (x_max_values[1:] - x_max_values[:-1]) / (t[1:] - t[:-1])

# Moyenne des positions (pour les placer entre x_i et x_{i+1})
x_moyennes = 0.5 * (x_max_values[1:] + x_max_values[:-1])

# Affichage de la vitesse en fonction de la position
plt.figure()
plt.plot(x_moyennes, v_propag)
plt.xlabel("Position x (m)")
plt.ylabel("Vitesse de propagation de la vague (m/s)")
plt.title("Vitesse de la vague en fonction de la position")
plt.grid()
plt.show()
"""
######################################################## Question 5.4 (d) ####################################################
"""

ecrire_valeur("fac","false")
ecrire_valeur("L",1000000)
ecrire_valeur("hR",20.0)
ecrire_valeur("hL",8000.0)
ecrire_valeur("xa",450000)
ecrire_valeur("xb",950000)
ecrire_valeur("cb_gauche","sortie")
ecrire_valeur("cb_droite","sortie")
ecrire_valeur("x1",50000)
ecrire_valeur("x2",250000)
ecrire_valeur("v_uniform","false")
ecrire_valeur("impose_nsteps","false")
ecrire_valeur("nx",1000)
ecrire_valeur("nsteps",64)
ecrire_valeur("CFL",1)
ecrire_valeur("equation_type","B")
ecrire_valeur("tfin",12000)
ecrire_valeur("initialization","autre")
ecrire_valeur("initial_state","right")
ecrire_valeur("f_hat",1.0)
ecrire_valeur("n_stride",10)


# Rapprocher les points
ecrire_valeur("xa",450000)

output_file = f"xa_450e3.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} xa_450e3 output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_xa_450e3 = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_xa_450e3.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (statique)")
plt.grid(True)
plt.tight_layout()

fig_xa_450e3, ax_xa_450e3 = plt.subplots()
line_xa_450e3, = ax_xa_450e3.plot([], [], lw=2)
ax_xa_450e3.set_xlim(x[0], x[-1])
ax_xa_450e3.set_ylim(np.min(f_xa_450e3), np.max(f_xa_450e3) * 1.1)
ax_xa_450e3.set_xlabel("x (m)")
ax_xa_450e3.set_ylabel("f(x, t)")
ax_xa_450e3.set_title("Animation vague")

def init_xa_450e3():
    line_xa_450e3.set_data([], [])
    return line_xa_450e3,

def update_xa_450e3(frame):
    line_xa_450e3.set_data(x, f_xa_450e3[frame, :])
    ax_xa_450e3.set_title(f"Statique, t = {frame*dt:.2f} s")
    return line_xa_450e3,

ani_static = animation.FuncAnimation(fig_xa_450e3, update_xa_450e3, frames=nt, init_func=init_xa_450e3,
                                     blit=True, interval=30)

plt.tight_layout()

# Vague initialement orientée vers la gauche
ecrire_valeur("xa",650000)
ecrire_valeur("tfin",12000)

output_file = f"xa_650e3.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} xa_650e3 output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_xa_650e3 = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_xa_650e3.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (gauche)")
plt.grid(True)
plt.tight_layout()

fig_xa_650e3, ax_xa_650e3 = plt.subplots()
line_xa_650e3, = ax_xa_650e3.plot([], [], lw=2)
ax_xa_650e3.set_xlim(x[0], x[-1])
ax_xa_650e3.set_ylim(np.min(f_xa_650e3), np.max(f_xa_650e3) * 1.1)
ax_xa_650e3.set_xlabel("x (m)")
ax_xa_650e3.set_ylabel("f(x, t)")
ax_xa_650e3.set_title("Animation vague vers la gauche")

def init_xa_650e3():
    line_xa_650e3.set_data([], [])
    return line_xa_650e3,

def update_xa_650e3(frame):
    line_xa_650e3.set_data(x, f_xa_650e3[frame, :])
    ax_xa_650e3.set_title(f"Gauche, t = {frame*dt:.2f} s")
    return line_xa_650e3,

ani_left = animation.FuncAnimation(fig_xa_650e3, update_xa_650e3, frames=nt, init_func=init_xa_650e3,
                                   blit=True, interval=30)
plt.tight_layout()


# Vague initialement orientée vers la droite
ecrire_valeur("xa",900000)
ecrire_valeur("tfin",12000)

output_file = f"xa_900e3.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} xa_900e3 output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_xa_900e3 = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_xa_900e3.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (droite)")
plt.grid(True)
plt.tight_layout()

fig_xa_900e3, ax_xa_900e3 = plt.subplots()
line_xa_900e3, = ax_xa_900e3.plot([], [], lw=2)
ax_xa_900e3.set_xlim(x[0], x[-1])
ax_xa_900e3.set_ylim(np.min(f_xa_900e3), np.max(f_xa_900e3) * 1.1)
ax_xa_900e3.set_xlabel("x (m)")
ax_xa_900e3.set_ylabel("f(x, t)")
ax_xa_900e3.set_title("Animation vague vers la droite")

def init_xa_900e3():
    line_xa_900e3.set_data([], [])
    return line_xa_900e3,

def update_xa_900e3(frame):
    line_xa_900e3.set_data(x, f_xa_900e3[frame, :])
    ax_xa_900e3.set_title(f"Droite, t = {frame*dt:.2f} s")
    return line_xa_900e3,

ani_right = animation.FuncAnimation(fig_xa_900e3, update_xa_900e3, frames=nt, init_func=init_xa_900e3,
                                    blit=True, interval=30)
plt.tight_layout()


# Afficher toutes les figures en même temps
plt.show()
"""

######################################################## Question 5.4 (e) ####################################################
"""
# Exemple pour la solution numérique et anlytique
g  = 9.81

ecrire_valeur("fac","false")
ecrire_valeur("L",1000000)
ecrire_valeur("hR",20.0)
ecrire_valeur("hL",8000.0)
ecrire_valeur("xa",450000)
ecrire_valeur("xb",950000)
ecrire_valeur("cb_gauche","sortie")
ecrire_valeur("cb_droite","sortie")
ecrire_valeur("x1",50000)
ecrire_valeur("x2",250000)
ecrire_valeur("v_uniform","false")
ecrire_valeur("impose_nsteps","false")
ecrire_valeur("nx",1000)
ecrire_valeur("nsteps",64)
ecrire_valeur("CFL",1)
ecrire_valeur("tfin",12000)
ecrire_valeur("initialization","autre")
ecrire_valeur("initial_state","right")
ecrire_valeur("f_hat",1.0)
ecrire_valeur("n_stride",10)


ecrire_valeur("equation_type","A")

output_file = f"Tsunami_A.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Tsunami_A output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_tsu_A = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_tsu_A.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (statique)")
plt.grid(True)
plt.tight_layout()

fig_tsu_A, ax_tsu_A = plt.subplots()
line_tsu_A, = ax_tsu_A.plot([], [], lw=2)
ax_tsu_A.set_xlim(x[0], x[-1])
ax_tsu_A.set_ylim(np.min(f_tsu_A), np.max(f_tsu_A) * 1.1)
ax_tsu_A.set_xlabel("x (m)")
ax_tsu_A.set_ylabel("f(x, t)")
ax_tsu_A.set_title("Animation vague")

def init_tsu_A():
    line_tsu_A.set_data([], [])
    return line_tsu_A,

def update_tsu_A(frame):
    line_tsu_A.set_data(x, f_tsu_A[frame, :])
    ax_tsu_A.set_title(f"Statique, t = {frame*dt:.2f} s")
    return line_tsu_A,

ani_static = animation.FuncAnimation(fig_tsu_A, update_tsu_A, frames=nt, init_func=init_tsu_A,
                                     blit=True, interval=30)

plt.tight_layout()

# Vague initialement orientée vers la gauche
ecrire_valeur("equation_type","C")

output_file = f"Tsunami_C.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Tsunami_C output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')

f_tsu_C = np.loadtxt("f_"+output_file)[:, 1:]
x = np.loadtxt("x_"+output_file)
E = np.loadtxt("en_"+output_file)[:, 1:]

nt, nx = f_tsu_C.shape
dx = x[1] - x[0]
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie (gauche)")
plt.grid(True)
plt.tight_layout()

fig_tsu_C, ax_tsu_C = plt.subplots()
line_tsu_C, = ax_tsu_C.plot([], [], lw=2)
ax_tsu_C.set_xlim(x[0], x[-1])
ax_tsu_C.set_ylim(np.min(f_tsu_C), np.max(f_tsu_C) * 1.1)
ax_tsu_C.set_xlabel("x (m)")
ax_tsu_C.set_ylabel("f(x, t)")
ax_tsu_C.set_title("Animation vague vers la gauche")

def init_tsu_C():
    line_tsu_C.set_data([], [])
    return line_tsu_C,

def update_tsu_C(frame):
    line_tsu_C.set_data(x, f_tsu_C[frame, :])
    ax_tsu_C.set_title(f"Gauche, t = {frame*dt:.2f} s")
    return line_tsu_C,

ani_left = animation.FuncAnimation(fig_tsu_C, update_tsu_C, frames=nt, init_func=init_tsu_C,
                                   blit=True, interval=30)
plt.tight_layout()
plt.show()
"""
##############################################################################################################################
##############################################   Facultatif  #################################################################
"""

ecrire_valeur("fac","true")
ecrire_valeur("equation_type","B") #Attention la C ne marche pas : problème dans l'éq.
ecrire_valeur("L",25.0)
ecrire_valeur("h00",1.0)
ecrire_valeur("cb_gauche","fixe")
ecrire_valeur("cb_droite","excitation")
ecrire_valeur("cb_bas","excitation")
ecrire_valeur("cb_haut","fixe")
ecrire_valeur("f_hat",0.0)
ecrire_valeur("x1",10.0)
ecrire_valeur("x2",15.0)
ecrire_valeur("v_uniform","true")
ecrire_valeur("nx",100)
ecrire_valeur("nsteps",16000)
ecrire_valeur("impose_nsteps","true")
ecrire_valeur("tfin",10.0)
ecrire_valeur("initialization","autre")
ecrire_valeur("n_stride",80)
ecrire_valeur("A",2.0)
ecrire_valeur("om",2.0)

# Vague initialement statique
ecrire_valeur("initial_state","right")

output_file = f"Question_fac.txt"
outputs.append(output_file)
cmd = f"./{executable} {input_filename} Question_fac output={output_file}"
print(cmd)
subprocess.run(cmd, shell=True)
print('Simulation terminée.')
actualise_valeur()

# Lecture des données spatiales
x = np.loadtxt("x_"+output_file)
time = np.loadtxt("f_"+output_file)[:,0]
dt = time[1]-time[0]
X, Y = np.meshgrid(np.linspace(0,L,nx+1), np.linspace(0,L,nx+1))

# Lecture du fichier f avec saut de ligne par ligne spatiale
f_all = np.loadtxt("f_"+output_file)[:,1:]

f_all = np.loadtxt("f_"+output_file)[:,1:]
print(f_all.shape)

# Reshape en (nt, nx, ny)
f_static = f_all.reshape(((nsteps)//n_stride+1, nx+1, nx+1))

# Figure et axe 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Initialisation de la surface
surf = ax.plot_surface(X, Y, f_static[0], cmap='viridis')

# Config axes
ax.set_xlim(x[0], x[-1])
ax.set_ylim(x[0], x[-1])
ax.set_zlim(np.min(f_static), np.max(f_static) * 1.1)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("f(x, y, t)")
ax.set_title("Animation f(x, y, t) - Statique")

# Fonction d'initialisation
def init():
    global surf
    surf.remove()
    #surf = ax.plot_surface(X, Y, f_static[0], cmap='viridis')
    surf = ax.plot_surface(X, Y, f_static[0],color ="blue")
    return surf,

# Fonction de mise à jour pour chaque frame
def update(frame):
    global surf
    surf.remove()
    #surf = ax.plot_surface(X, Y, f_static[frame], cmap='viridis')
    surf = ax.plot_surface(X, Y, f_static[frame],color ="blue")
    ax.set_title(f"Vague statique, t = {frame*dt:.2f} s")
    return surf,

# Animation
ani = animation.FuncAnimation(fig, update, frames=nsteps, init_func=init,
                              blit=False, interval=50)

plt.tight_layout()
plt.show()
"""