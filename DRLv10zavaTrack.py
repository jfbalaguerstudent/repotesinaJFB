import numpy as np
from collections import deque, namedtuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import sample
import numpy.random as rand 
import control as co
import control.matlab as coa
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy import stats

from collections import deque, namedtuple
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam



################## HYPERPARAMETERS ##################
NAMELY = "DRLv10zavaTrack_"


### Environment
OBSER_SIZE  = 4
TRACK_LEN   = 100                        # num of observations in a state
STATE_SIZE  = OBSER_SIZE * TRACK_LEN    # num of features in a state
Ts          = 0.001                     # Sample rate in seconds
aTs         = 10                        # Ammount of time steps for action hold 
NEPIS       = 500000                    # Number of maximum episodes
T           = 100.0                     # Max time of an episode
MAX_STEPS   = int(T/Ts)                 # Max number of steps in an episode
R_PUMP      = 10                        # Reward sensitivity to energy generated


### Agent
HIDDEN_SIZE = 256                       # Number of neurons in the hidden layers
N_a         = 9                         # Number of discrete actions
U_side      = 10                        # Max action in either direction
SIGMA       = 1                         # deviation of the normal distribution of the action space
GAM         = 0.993                     # Discount factor
EPSi        = 1.0                       # Initial exploration prob
ALPHA       = 1e-6                      # Step size of the V function update
BATCH_SIZE  = 256                       # Mini-batch size for experience replay
BUFF_SIZE   = int(5e5)                  # Replay buffer size
nSteps4new  = 10                        # steps before starting the updates
STRAP_TIME  = 150000                    # initial steps before updating the target network
patience    = 10.0                      # patience for copying the weights
LAM         = 881                       # trace length

### Processing
AVER_WINDOW = 100



Qnetwork = Sequential([
    Input(shape = (STATE_SIZE,)),
    Dense(units = HIDDEN_SIZE, activation = 'relu'),
    Dense(units = HIDDEN_SIZE, activation = 'relu'),
    Dense(units = HIDDEN_SIZE, activation = 'relu'),
    Dense(units = N_a)    # action space
])


mirrorQnet = Sequential([
    Input(shape = (STATE_SIZE,)),
    Dense(units = HIDDEN_SIZE, activation = 'relu'),
    Dense(units = HIDDEN_SIZE, activation = 'relu'),
    Dense(units = HIDDEN_SIZE, activation = 'relu'),
    Dense(units = N_a)    # action space
])

optimizer = Adam(learning_rate = ALPHA)






class wavestar:
    """
    This environment class needs the following libs:
import numpy as np
import math as m
import matlab.engine
    """

    def __init__(self):
        self.x = np.zeros((6,1))
        self.xd = np.zeros((6,1))
        self.fex = np.empty((1,0))
        self.n = 0
        self.T = T
        self.v = np.zeros((2,1))
        self.I = 0
        self.u = 0
        assert(Ts == 0.001)
        self.A = np.array([
            [998.522667876331e-3,    6.79382709256346e-3,    -784.223743194067e-6,   616.309527037955e-6,    -6.52297849396041e-6,   -1.23998031717525e-006],
            [-6.73304019537635e-3,   998.449980352090e-3,    -1.85358629035471e-3,   -1.20218724804023e-3,   -80.8935715444288e-6,   -395.128439090171e-006],
            [-78.9001974253207e-6,  -1.61828509819246e-3,     995.681020259407e-3,    6.14977405750622e-3,    -14.2342617765268e-6,   -5.84660826937975e-006],
            [1.41559003968001e-3,   -333.329607520467e-6,    -6.12185998701912e-3,   995.743006898599e-3,    -22.8357650576410e-6,   -135.276232077296e-006],
            [20.5676125303973e-6,    13.0940289432283e-6,    -4.53042389504218e-6,   -65.8768620744504e-6,   998.774715394314e-3,    12.3981639639189e-003],
            [-10.4947744622523e-6,  -6.83761090970159e-6,     40.9148090704912e-6,    -7.48979366876991e-6,   -12.3981793316588e-3,   998.774677649749e-003]
        ])
        self.B = np.array([
            [58.7101578364411e-6],
            [1.04482477821805e-3],
            [-78.5173456148556e-6],
            [368.245534915228e-6],
            [48.1443459470551e-6],
            [174.389566250718e-6]
        ])
        self.C = np.array([-153.996431865825e-3, 702.064666021535e-3, 16.3672129722999e-3, 102.420910774998e-3, -76.5203903945289e-3, -126.027549778853e-3])

        self.Ad = np.array([
            [0.986099977997927,  -0.019166252303374,  -0.010399496618668,  -0.011566214487255,  -0.005948397687302,  -0.006944942705682],
            [0.015888952153603,   0.999846535693164,  -0.000083265820251,  -0.000092681681227,  -0.000047660896632,  -0.000055689154859],
            [0.000127408152167,   0.015999181166673,   0.999999555730114,  -0.000000494706911,  -0.000000254387641,  -0.000000297354177],
            [0.000000340150127,   0.000063998361906,   0.007999999111238,   0.999999999010101,  -0.000000000509010,  -0.000000000595123],
            [0.000000000680774,   0.000000170664045,   0.000031999998578,   0.007999999998416,   0.999999999999185,  -0.000000000000953],
            [0.000000000000545,   0.000000000170665,   0.000000042666666,   0.000015999999999,   0.004,               1.0]
        ])
        self.Bd = np.array([
            0.003972238038401,
            0.000031852038042,
            0.000000170075064,
            0.000000000340387,
            0.000000000000545,
            0.0
        ])
        self.Cd = np.array([ -0.739600897734554,  -0.993530585169879,  -0.965242508196239,  -1.478561026117339,  -1.100688663513888,  -1.288103697663179])
        self.Dd = np.array([0.736745387557548])


    def step(self,u):
        self.x = np.dot(self.A,self.x) + self.B*(u+self.fex[self.n])
        y = np.dot(self.C,self.x)

        self.xd = np.dot(self.Ad,self.xd) + self.Bd*(u+self.fex[self.n])
        yd = np.dot(self.Cd,self.xd) + self.Dd*(u+self.fex[self.n])
        

        self.v[1,0] = self.v[0,0]
        self.v[0,0] = y[0] 

        self.I = self.I + Ts*(self.v[0,0]+self.v[1,0])/2


        r = -y[0]*u*Ts*R_PUMP

        self.n+=1

        if self.n > self.T/Ts:
            isDone = True
        else:
            isDone = False
        
        if np.abs(self.v[0,0]) > 15:
            isGone = True
            r = r - 1
        else:
            isGone = False

        s = np.concatenate(
            np.array([[self.v[0,0]], [self.I], [yd[0]], [self.u]],dtype=np.float32)
        )
        self.u = u

        return s, r, isDone, isGone 


    def reset(self):
        self.x = np.zeros((6,1))
        self.xd = np.zeros((6,1))
        self.I = 0
        self.n = 0
        self.v = np.zeros((2,1))
        self.u = 0
        return self.v[0,0]

    def JONSWAP(self, w, Tp, Hs, gamma):
        nw = len(w)
        f = w / (2 * np.pi)
        A = 0.3125 * Hs ** 2 / Tp ** 4
        B = 1.25 / Tp ** 4
        fp = 1. / Tp
        m0 = 0.
        m_1 = 0.
        Pwave = 0.
        Hw = np.zeros(nw)
        S = np.zeros(nw)

        for i in range(1, nw):
            fc = 0.5 * (f[i] + f[i - 1])
            df = f[i] - f[i - 1]
            S[i] = A / fc ** 5 * np.exp(-B / fc ** 4)
            sigma = 0.02 if fc < fp else 0.25
            pa = np.exp(-(fc - fp) ** 2 / (2. * sigma ** 2 * fp ** 2))
            S[i] *= gamma ** pa
            m0 += S[i] * df
            m_1 += S[i] / fc * df

        alpha = Hs ** 2 / (16. * m0)
        Te = m_1 / m0

        for i in range(1, nw):
            df = f[i] - f[i - 1]
            S[i] *= alpha
            Hw[i] = np.sqrt(2. * S[i] * df)
            Pwave += 0.25 * 1025. * 9.81 * 9.81 * Hw[i] * Hw[i] / w[i]

        return S, Hw, Pwave, Te

    # --- Spectrum of signal ---
    def espectro(self, y, Fs):
        T = 1 / Fs
        L = len(y)
        t = np.arange(0, L) * T
        NFFT = 2 ** int(np.ceil(np.log2(L)))
        Y = fft(y, NFFT) / L
        Y = Y[:NFFT // 2 + 1]
        f = Fs / 2 * np.linspace(0, 1, NFFT // 2 + 1)
        return t, f, Y

    # --- Inverse FFT to reconstruct time signal ---
    def fftInvertion(self, Fspe, dt, t_max):
        Nl = int(t_max / dt)
        f_fSt = Fspe.T
        if_fSt = np.concatenate([f_fSt[:-1], f_fSt[-2:0:-1]])
        rSt = np.real(ifft(if_fSt, norm='forward'))

        timeinv = np.arange(0, len(rSt)) * dt
        ddd = np.argmin(np.abs(timeinv - t_max))
        rSt = rSt[:ddd - 1] * Nl
        timeinv = timeinv[:ddd - 1]
        return rSt, timeinv

    # --- Generate wave and excitation force ---
    def whitenoiseWave(self, dt, t_max, Hs, Tw, s, wfreq, Fesys, DD):
        Fs = 1 / dt
        Nl = int(np.ceil(t_max * Fs))

        np.random.seed(DD)
        nin = s * Hs * np.random.randn(Nl)
        _, fnin, NIN = self.espectro(nin, Fs)
        _, Hs2i, _, _ = self.JONSWAP(2 * np.pi * fnin, Tw, Hs, 3.3)

        interp_func = interp1d(wfreq, Fesys, kind='linear', bounds_error=False, fill_value=0)
        WECFe = interp_func(2 * np.pi * fnin)

        Fspe = NIN * Hs2i * WECFe
        Wspe = NIN * Hs2i

        waveG, timevec = self.fftInvertion(Wspe, dt, t_max)
        forcG, _ = self.fftInvertion(Fspe, dt, t_max)
        cte = Hs / (4 * np.std(waveG))
        return waveG * cte, forcG * cte, timevec, fnin, Hs2i
    
    def loadw2f(self):
        # Carga manual del modelo de la ola a la fuerza (W2F)
        A = np.array([[-3.949,5.07,2.104e-05,6.351e-06],[-1.267,-3.949,-3.017,3.334],[-9.698e-06,-2.511e-06,-3.649,8.92],[3.649e-06,-2.18e-05,-8.92,-3.649]])
        B = np.array([[1.485e-05],[34.89],[3.134e-06],[33]])
        C = np.array([44.9,-19.32,1.953e-07,7.14e-06])
        system = coa.ss(A,B,C,0)
        return system

    def loadf2v(self):
        # Carga manual del modelo de fuerza a velocidad (F2V)
        # Matrix A
        A = np.array([
        [-1.456, 6.804, -0.7782, 0.6246, -0.00625, 0.0001826],
        [-6.742, -1.53, -1.865, -1.198, -0.08351, -0.3952],
        [-0.08898, -1.622, -4.311, 6.175, -0.0143, -0.005676],
        [1.418, -0.3441, -6.148, -4.248, -0.02379, -0.1356],
        [0.02075, 0.01307, -0.00498, -0.06599, -1.149, 12.41],
        [-0.0104, -0.006698, 0.04096, -0.008047, -12.41, -1.149]
        ])

        # Matrix B
        B = np.array([
        [0.05505],
        [1.046],
        [-0.07897],
        [0.3689],
        [0.04709],
        [0.1748]
        ])

        # Matrix C
        C = np.array([
        [-0.154, 0.7021, 0.01637, 0.1024, -0.07652, -0.126]
        ])

        # Matrix D
        D = np.array([
        [0]
        ])
        system = coa.ss(A,B,C,D)
        return system
    



    def waveGen(self, seed,Hs,Tw):
        dt = Ts   # Paso
        t_end = 200.0# Tiempo de simulación

        W2F_model = self.loadw2f()
        W2F_advance = 0.46

        w = np.arange(0, t_end+Ts, Ts)
        Fe,_,_ = co.frequency_response(W2F_model, w)
        Fe = Fe.squeeze().T * np.exp(1j * w * W2F_advance)

        Hw = Hs
        #Tw = 1.3
        #gamma = 3.3
        #seed_realisation = seed  # Semilla

        eta, Fex, tsim, _, _ = self.whitenoiseWave(dt, t_end, Hw, Tw, 1, w, Fe, seed)
        
        #Fex = np.array(Fex)
        #Fex = Fex[0]

        self.fex = Fex
        
        return
    
    def sinGen(self,Hs, Tw):
        dt = Ts
        t_end = T + 1.0
        t = np.arange(0, t_end, dt)
        omega = 2 * np.pi / Tw
        phase = np.random.uniform(0, 2*np.pi)
        eta = Hs * np.sin(omega * t + phase)

        self.fex = eta

        return
        




def getLoss(exps, gams, LAM, Qnet, sameQnet):
    s, a, RnN, s_N = exps    #unfold

    nextQsa = tf.reduce_max(sameQnet(s_N), axis=-1)

    #set y = R if episode terminates, otherwise set y = R + gamma max[Q'(s,a)]
    RnN = tf.cast(RnN,tf.float32)
    y_targets = RnN + (gams**LAM)*nextQsa       ##!!!!!! it's a continuing problem, it should never stop adding the estimation at the next step
    #get the calues and reshape it to match the y_targets
    q_values = Qnet(s)
    q_values = tf.gather_nd(q_values, tf.stack([ tf.range(q_values.shape[0]),
                                                tf.cast(a, tf.int32)    ], axis=1))

    loss = MSE(y_targets, q_values)

    return loss

@tf.function
def Qlearn(exps, gams, LAM):

    with tf.GradientTape() as tape:
        loss = getLoss(exps, gams, LAM, Qnetwork, mirrorQnet)

    grads = tape.gradient(loss, Qnetwork.trainable_variables)

    optimizer.apply_gradients(zip(grads, Qnetwork.trainable_variables))

    return loss


def getA(eps, Qvalues) -> int:

    if np.random.random() < eps:
        return np.random.randint(0,N_a)
    else:
        return int(np.argmax(Qvalues))


def getExps(buff, NBATCH):
    exps = sample(buff, NBATCH)

    # Stack fields directly
    states     = np.stack([e.state for e in exps]).astype(np.float32)
    actions    = np.stack([e.action for e in exps]).astype(np.int32)
    rewards    = np.stack([e.reward for e in exps]).astype(np.float32)
    nextStates = np.stack([e.nextState for e in exps]).astype(np.float32)

    # Return same namedtuple structure
    mini = exps[0]._replace(
        state=states,
        action=actions,
        reward=rewards,
        nextState=nextStates
    )
    return mini


def checkUpdate(t, nSteps4new,  memBuff) -> bool:
    if len(memBuff) < BATCH_SIZE:
        return False
    else:
        if t%nSteps4new == 0:               # in n step TD learning, this might be irrelevant.
            return True
        else:
            return False
        
def newEpsilon(eps):
    return max(0.03, eps*0.999)



def watchme(idx):
    t = np.linspace(0,Ts*MAX_STEPS,MAX_STEPS)

    Fe = ws.fex

    u = np.zeros((np.size(t,0)))
    y = np.zeros((np.size(t,0)))
    r = np.zeros((np.size(t,0)))

    y[0] = ws.reset()

    state = np.zeros((TRACK_LEN,OBSER_SIZE), dtype=np.float32)

    for k in range(1,MAX_STEPS):

        stateQn = state.reshape(1,-1).astype(np.float32)
        Qvals = Qnetwork(stateQn)
        action = getA(eps, Qvals)
        alin = (2*action-N_a+1)/(N_a-1)
        uQ = U_side * (alin * np.exp((alin/SIGMA)**2))/np.exp(1/SIGMA**2)
        

        for l in range(aTs):
            u[aTs*k-1+l] = uQ
            nextObser, r[aTs*k+l], isDone, isGone = ws.step(uQ)

            r[aTs*k+l] = r[aTs*k+l]/R_PUMP + r[aTs*k-1+l]

            isDone = isDone or isGone

            y[aTs*k+l] = nextObser[0]
            if isDone:
                break

        state[1:,:] = state[:-1,:]
        state[0,:]  = np.asarray(nextObser, dtype=np.float32).reshape(-1)


        if aTs*k == MAX_STEPS-aTs or isDone:
            try:
                fig, ax = plt.subplots()
                ax.plot(t[:aTs*k],u[:aTs*k], color='C8',linewidth=0.5)
                ax.plot(t[:aTs*k],Fe[:aTs*k], color='C0')
                ax.plot(t[:aTs*k],y[:aTs*k], color='C1')
                ax.plot(t[:aTs*k],r[:aTs*k],color='C2')
                ax.legend(["a","Fe","v","R"])
                ax.grid()

                name = "./" + NAMELY + idx + ".pdf"
            
                fig.savefig(name)
                #plt.show()
                # arbitrary space here?
                plt.close('all')
            except:
                print("Could not print :(")


            try:
                fig, ax = plt.subplots()
                ax.plot(t[:int(10//Ts)],u[:int(10//Ts)], color='C8',linewidth=0.5)
                ax.plot(t[:int(10//Ts)],Fe[:int(10//Ts)], color='C0')
                ax.plot(t[:int(10//Ts)],y[:int(10//Ts)], color='C1')
                ax.plot(t[:int(10//Ts)],r[:int(10//Ts)],color='C2')
                ax.legend(["a","Fe","v","R"])
                ax.grid()

                name = "./" + NAMELY + idx + "ZOOM.pdf"
            
                fig.savefig(name)
                #plt.show()
                # arbitrary space here?
                plt.close('all')
            except:
                print("Could not print :(")

            return
    return

















#### Process Object ####
ws = wavestar()
#ws.sinGen(5,1.3)
ws.waveGen(1,0.15/2,1.3)
ws.reset()

########################
########################




print()
print()

#watchme("0")

copyTime = STRAP_TIME
lastStrap = 0
onstrap = np.float32(0.0)
eps = EPSi
niter = 0
iter = []
miter = []
lost = []
mlost = []
loston = []
gams = GAM
rc = []
epistrap = []
pointHist = []
averageLastPts = []
goodCount = 0


memBuff = deque(maxlen = BUFF_SIZE)

Experience = namedtuple("Experience", field_names=["state","action","reward","nextState"])


mirrorQnet.set_weights(Qnetwork.get_weights())


for i in range(NEPIS):
    ws.reset()
    obser = np.zeros(OBSER_SIZE, dtype=np.float32)
    state = np.zeros((TRACK_LEN, OBSER_SIZE), dtype=np.float32)
    totalPts = 0
    didstrap = False
    story = np.empty((0, STATE_SIZE), dtype=np.float32)
    route = []
    loosen = []
    rs = []

    for t in range(MAX_STEPS):

        stateQn = state.reshape(1,-1).astype(np.float32)
        Qvals = Qnetwork(stateQn)
        action = getA(eps, Qvals)
        route.append(action)

        alin = (2*action-N_a+1)/(N_a-1)
        u = U_side * (alin * np.exp((alin/SIGMA)**2))/np.exp(1/SIGMA**2)

        rp = 0
        for _ in range(aTs):
            nextObser, r, isDone, isGone = ws.step(u)
            rp += r
        
            isDone = isDone or isGone

            if isDone:
                break
        rs.append(rp)
        state_flat = state.reshape(1, -1).astype(np.float32)
        story = np.vstack([story, state_flat])

        state[1:,:] = state[:-1,:]
        state[0,:]  = np.asarray(nextObser, dtype=np.float32).reshape(-1)

        totalPts += rp/R_PUMP

        update = checkUpdate(t, nSteps4new,  memBuff)

        loose = 0.0
        if update:
            experiences = getExps(memBuff, BATCH_SIZE)
            loose = Qlearn(experiences,gams,LAM)
            lost.append(loose.numpy())
            loosen.append(loose.numpy())
            iter.append(niter)
            niter += 1
            if len(lost) > 30000:
                lost.pop(0)
                iter.pop(0)
                try:
                    if miter[0] < iter[0]:
                        miter.pop(0)
                except:
                    pass
            if len(mlost) > AVER_WINDOW:
                mlost.append(np.average(lost[len(mlost)-AVER_WINDOW:len(mlost)]))
                if len(mlost) > 30000:
                    mlost.pop(0)
            else:
                mlost.append(np.average(lost))
            
            if niter - lastStrap >= copyTime:
                mirrorQnet.set_weights(Qnetwork.get_weights())
                didstrap = True
                onstrap = np.float32(1.0)
                miter.append(niter)
                lastStrap = niter
                copyTime = max(copyTime*0.9, int(5/Ts))
            try:
                if mlost[-1] > patience:
                    copyTime = STRAP_TIME
            except:
                pass
        


        if isDone:
            break
    
    for m in range(len(rs)-LAM):
        RnN = sum([rs[m+l]*(gams**l) for l in range(LAM)])
        memBuff.append(Experience(story[m,:],route[m],RnN,story[m+LAM,:]))
    
    pointHist.append(totalPts)
    averageLastPts.append(np.mean(pointHist[-AVER_WINDOW:]))
    loston.append(np.average(loosen) if len(loosen) > 0 else 1.0)
    if didstrap:
        epistrap.append(i)

    eps = newEpsilon(eps)

    print(f"\rEpisode {i+1} | epsilon: {eps:.4f} | Total point average of the last {AVER_WINDOW} episodes: {averageLastPts[-1]:.2f}| Loss on the last {AVER_WINDOW} episodes: {np.average(loston[-AVER_WINDOW:] if len(loston) > 1 else 1.0):.4f} ",end="")

    #ws.sinGen(5,1.3)
    ws.waveGen(i,0.15/2,1.3)

    if (i+1) % AVER_WINDOW == 0:#(i+1)/5 % averWindow == 0:

        figLo, axLo = plt.subplots()
        axLo.axhline(y=1, linewidth=0.2, color='k')
        for vline in epistrap:
            axLo.axvline(x=vline, color='red',linewidth=0.5)
        axLo.semilogy(loston, linewidth=0.5)
        axLo.set_xlabel(r"\bf{Episode}")
        axLo.set_ylabel(r"\bf{Lost History}")
        axLo.legend([r"$ref$", r"$Lost on epi.$"])
        name = "./" + NAMELY + str((i+1)//AVER_WINDOW) + "loston.pdf"
        figLo.savefig(name)
        #except:
        #    print("image lost :(")
        #plt.show()
        # arbitrary space here?
        plt.close('all')

        figLoss, axLoss = plt.subplots()
        for vline in miter:
            axLoss.axvline(x=vline, color='red',linewidth=0.5)
        axLoss.axhline(y=1, linewidth=0.2, color='k')
        axLoss.semilogy(iter,lost, linewidth=0.5)
        axLoss.semilogy(iter,mlost)                                          ############################## Watch out!!! Peligro de derrumbe
        axLoss.set_xlabel(r"\bf{Iteraciones}")
        axLoss.set_ylabel(r"\bf{error $J$}")
        axLoss.legend([r"$J$", r"$\bar{J}$"])
        name = "./" + NAMELY + str((i+1)//AVER_WINDOW) + "loss.pdf"
        figLoss.savefig(name)
        #except:
        #    print("image lost :(")
        #plt.show()
        # arbitrary space here?
        plt.close('all')

        figRet, axRet = plt.subplots()
        axRet.axhline(y=0, linewidth=0.2, color='k')
        axRet.plot(pointHist, linewidth=0.5)
        axRet.plot(averageLastPts)
        axRet.set_xlabel(r"\bf{Iteraciones}")
        axRet.set_ylabel(r"\bf{Point History}")
        axRet.legend([r"$ref$", r"$Return$"])
        name = "./" + NAMELY + str((i+1)//AVER_WINDOW) + "Ret.pdf"
        figRet.savefig(name)
        #except:
        #    print("image lost :(")
        #plt.show()
        # arbitrary space here?
        plt.close('all')



    if (i+1) % AVER_WINDOW == 0:
        print(f"\rEpisode {i+1} | epsilon: {eps:.4f} | Total point average of the last {AVER_WINDOW} episodes: {averageLastPts[-1]:.2f}| Loss on the last {AVER_WINDOW} episodes: {np.average(loston[-AVER_WINDOW:] if len(loston) > 1 else 1.0):.4f}")
        with open("training_log.txt", "a") as log_file:
            log_file.write(f"Episode {i+1} | epsilon: {eps:.4f} | Total point average of the last {AVER_WINDOW} episodes: {averageLastPts[-1]:.2f}| Loss on the last {AVER_WINDOW} episodes: {np.average(loston[-AVER_WINDOW:] if len(loston) > 1 else 1.0):.4f}\n")
        watchme(str((i+1)//AVER_WINDOW))
        #ws.waveGen(((i+1)//averWindow),0.3/2,1.3)
        name = './kerases/' + NAMELY + 'temp' + str((i+2//AVER_WINDOW)) + '.keras'
        Qnetwork.save(name)

