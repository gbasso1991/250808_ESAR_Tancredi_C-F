
#%%Comparador resultados de medidas ESAR en C-F de Pablo Tancredi
# Medidas a 300 kHz y 57 kA/m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
#%% Funciones
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)

    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()

            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos

            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m

            # Crear etiqueta para la leyenda
            nombre_base = os.path.split(archivo)[-1].split('_')[1]
            #os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"

            # Graficar

            ax.plot(campo, magnetizacion, label=etiqueta)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue

    plt.xlabel('Campo magnético (kA/m)')
    plt.ylabel('Magnetización (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()

def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=20,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% Localizo ciclos y resultados
ciclos_C1=glob(('C1/**/*ciclo_promedio*'),recursive=True)
ciclos_C1.sort()

res_C1=glob('C1/**/*resultados*',recursive=True)
res_C1.sort()
ciclos_C2 = glob('C2/**/*ciclo_promedio*', recursive=True)
ciclos_C2.sort()
res_C2 = glob('C2/**/*resultados*', recursive=True)
res_C2.sort()

ciclos_C4 = glob('C4/**/*ciclo_promedio*', recursive=True)
ciclos_C4.sort()
res_C4 = glob('C4/**/*resultados*', recursive=True)
res_C4.sort()

labels_C=['C1','C2','C3','C4']

ciclos_F1 = glob('F1/**/*ciclo_promedio*', recursive=True)
ciclos_F1.sort()
res_F1 = glob('F1/**/*resultados*', recursive=True)
res_F1.sort()

ciclos_F2 = glob('F2/**/*ciclo_promedio*', recursive=True)
ciclos_F2.sort()
res_F2 = glob('F2/**/*resultados*', recursive=True)
res_F2.sort()

ciclos_F3 = glob('F3/**/*ciclo_promedio*', recursive=True)
ciclos_F3.sort()
res_F3 = glob('F3/**/*resultados*', recursive=True)
res_F3.sort()

ciclos_F4 = glob('F4/**/*ciclo_promedio*', recursive=True)
ciclos_F4.sort()
res_F4 = glob('F4/**/*resultados*', recursive=True)
res_F4.sort()

labels_F=['F1','F2','F3','F4']

#%% comparo los ciclos C 
plot_ciclos_promedio('C1')
plot_ciclos_promedio('C2')
plot_ciclos_promedio('C3')
plot_ciclos_promedio('C4')
#%% comparo los ciclos F
plot_ciclos_promedio('F1')
plot_ciclos_promedio('F2')
plot_ciclos_promedio('F3')
plot_ciclos_promedio('F4')
#%%
f1-1- 123146
f2 -3 - 124049
f3 - 1 - 124517
f4 - 1 - 125106
#%%
#%% selecciono 
C1-2 -115711
c2-1 -120233
c3-2 -121118
c4-2 121712 
#%%

# Patrón que busca en todas las subcarpetas de C1, filtra por "ciclo_promedio" y luego por "115711" en la ruta
ciclo_C1 = glob('C1/**/*115711/**/*ciclo_promedio*', recursive=True)
ciclo_C2 = glob('C2/**/*120233/**/*ciclo_promedio*', recursive=True)
ciclo_C3 = glob('C3/**/*121118/**/*ciclo_promedio*', recursive=True)
ciclo_C4 = glob('C4/**/*121712/**/*ciclo_promedio*', recursive=True)
ciclo_F1 = glob('F1/**/*123146/**/*ciclo_promedio*', recursive=True)
ciclo_F2 = glob('F2/**/*124049/**/*ciclo_promedio*', recursive=True)
ciclo_F3 = glob('F3/**/*124517/**/*ciclo_promedio*', recursive=True)
ciclo_F4 = glob('F4/**/*125106/**/*ciclo_promedio*', recursive=True)

# Leer y graficar los ciclos seleccionados para muestras C y F

fig, (a, b) = plt.subplots(2, 1, figsize=(8, 9), sharex=True, sharey=True, constrained_layout=True)

# Muestras C
ciclos_C = [ciclo_C1, ciclo_C2, ciclo_C3, ciclo_C4]
labels_C_plot = ['C1', 'C2', 'C3', 'C4']

for idx, (ciclo, label) in enumerate(zip(ciclos_C, labels_C_plot)):
    if ciclo:  # Verifica que la lista no esté vacía
        t, H_Vs, M_Vs, H_kAm, M_Am, metadata = lector_ciclos(ciclo[0])
        a.plot(H_kAm, M_Am, '.-', label=label)

a.set_ylabel('Magnetización (A/m)')
a.legend(ncol=1, loc='lower right')
a.grid()
a.set_title('Muestras C - 300 kHz - 57 kA/m')
a.set_xlabel('Campo (A/m)')

# Muestras F
ciclos_F = [ciclo_F1, ciclo_F2, ciclo_F3, ciclo_F4]
labels_F_plot = ['F1', 'F2', 'F3', 'F4']

for idx, (ciclo, label) in enumerate(zip(ciclos_F, labels_F_plot)):
    if ciclo:
        t, H_Vs, M_Vs, H_kAm, M_Am, metadata = lector_ciclos(ciclo[0])
        b.plot(H_kAm, M_Am, '.-', label=label)

b.set_ylabel('Magnetización (A/m)')
b.set_xlabel('Campo (A/m)')
b.legend(ncol=1, loc='lower right')
b.grid()
b.set_title('Muestras F - 300 kHz - 57 kA/m')

#plt.savefig('comparativa_ciclos_C_F.png', dpi=300)
plt.show()



# %%
#%%










#%% SAR
fig, (ax,ax2) = plt.subplots(nrows=2,figsize=(8,6),constrained_layout=True,sharex=True,sharey=True)
ax.set_title(f' {f_label/1000:.0f} kHz - {H_label/1000:.0f} kA/m - congelado sin Campo',loc='left',y=0.89)
ax.plot(T_csC1,SAR_csC1,'.-',c='C0',label=labels_sC[0])
ax.plot(T_csC2,SAR_csC2,'.-',c='C1',label=labels_sC[1])

ax2.set_title(f' {f_label/1000:.0f} kHz - {H_label/1000:.0f} kA/m - congelado con Campo',loc='left',y=0.89)
ax2.plot(T_ccC1,SAR_ccC1,'.-',c='C2',label=labels_cC[0])
ax2.plot(T_ccC2,SAR_ccC2,'.-',c='C3',label=labels_cC[1])

for a in fig.axes:
    a.grid()   
    a.set_ylabel('SAR (W/g)')
    a.legend(ncol=1)
ax2.set_xlabel('Temperature (°C)')
plt.suptitle('SAR vs T',fontsize=15)    
# ax.set_xlim(0,60e3)
# ax.set_ylim(0,)
#plt.savefig('8A_ciclo_HM_'+idc[i]+'.png',dpi=400)
plt.show()
#%% tau
fig, (ax,ax2) = plt.subplots(nrows=2,figsize=(8,6),constrained_layout=True,sharex=True,sharey=True)
ax.set_title(f' {f_label/1000:.0f} kHz - {H_label/1000:.0f} kA/m - congelado sin Campo',loc='left',y=0.89)
ax.plot(T_csC1,tau_csC1,'.-',c='C0',label=labels_sC[0])
ax.plot(T_csC2,tau_csC2,'.-',c='C1',label=labels_sC[1])

ax2.set_title(f' {f_label/1000:.0f} kHz - {H_label/1000:.0f} kA/m - congelado con Campo',loc='left',y=0.89)
ax2.plot(T_ccC1,tau_ccC1,'.-',c='C2',label=labels_cC[0])
ax2.plot(T_ccC2,tau_ccC2,'.-',c='C3',label=labels_cC[1])

for a in fig.axes:
    a.grid()   
    a.set_ylabel('tau (W/g)')
    a.legend(ncol=1,loc='lower left')
ax2.set_xlabel('Temperature (°C)')
plt.suptitle('tau vs T',fontsize=15)    
# ax.set_xlim(0,60e3)
# ax.set_ylim(0,)
#plt.savefig('8A_ciclo_HM_'+idc[i]+'.png',dpi=400)
plt.show()


# %%
