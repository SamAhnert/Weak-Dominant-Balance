'''FD operator specifically given specifically to accompany the turbulent duct flow data and meshgrid'''

import numpy as np

def diff6(f, x):
    a1 = 150/256
    a2 = -25/256
    a3 = 3/256
    b1 = 1/16
    b2 = 1/24
    c1 = 75/64
    c2 = -25/384
    c3 = 3/640
    d1 = 1.5
    d2 = -0.3
    d3 = 1/30

    nx = x.size

    Q = np.ndarray(nx)
    for i in range(2,nx-3):
        Q[i] = a1 * (f[i+1]+f[i]) + a2 * (f[i+2]+f[i-1]) + a3*(f[i+3]+f[i-2])

    Q0     =b1* (35 *f[0] - 35 *f[1]   + 21 *f[2]   - 5 *f[3])

    Q[0]   =b1* (5  *f[0] + 15 *f[1]   - 5  *f[2]   +    f[3])
    Q[1]   =b1* (-   f[0] + 9  *f[1]   + 9  *f[2]   -    f[3])
    Q[nx-3]=b1* (-   f[nx-1]+ 9  *f[nx-2]+ 9  *f[nx-3]-    f[nx-4])
    Q[nx-2]=b1* (5  *f[nx-1]+ 15 *f[nx-2]- 5  *f[nx-3]+    f[nx-4])
    Q[nx-1]=b1* (35 *f[nx-1]- 35 *f[nx-2]+ 21 *f[nx-3]- 5 *f[nx-4])

    
    fx =  np.ndarray(nx)
    for i in range(3, nx-3):
        fx[i] = (c1*(Q[i]-Q[i-1])+c2*(Q[i+1]-Q[i-2])+c3*(Q[i+2]-Q[i-3]))


    fx[0]   = b2* (-23  *Q0     +21*Q[0]   +3 *Q[1]   -  Q[2])
    fx[1]   = b2* (-22  *Q[0]   +17*Q[1]   +9 *Q[2]   -5*Q[3]   +Q[4])
    fx[2]   = b2* (      Q[0]   -27*Q[1]   +27*Q[2]   -  Q[3])
    fx[nx-3]=-b2* (      Q[nx-2]-27*Q[nx-3]+27*Q[nx-4]-  Q[nx-5])
    fx[nx-2]=-b2* (-22  *Q[nx-2]+17*Q[nx-3]+9 *Q[nx-4]-5*Q[nx-5]+Q[nx-6])
    fx[nx-1]=-b2* (-23  *Q[nx-1]+21*Q[nx-2]+3 *Q[nx-3]-  Q[nx-4])

    QQ = np.ndarray(nx)
    for i in range(2, nx-3):
        QQ[i] = a1*(x[i+1]+x[i])+a2*(x[i+2]+x[i-1])+a3*(x[i+3]+x[i-2])

    QQ0     =b1* (35 *x[0] - 35 *x[1]   + 21 *x[2]   - 5 *x[3])

    QQ[0]   =b1* (5  *x[0] + 15 *x[1]   - 5  *x[2]   +    x[3])
    QQ[1]   =b1* (-   x[0] + 9  *x[1]   + 9  *x[2]   -    x[3])
    QQ[nx-3]=b1* (-   x[nx-1]+ 9  *x[nx-2]+ 9  *x[nx-3]-    x[nx-4])
    QQ[nx-2]=b1* (5  *x[nx-1]+ 15 *x[nx-2]- 5  *x[nx-3]+    x[nx-4])
    QQ[nx-1]=b1* (35 *x[nx-1]- 35 *x[nx-2]+ 21 *x[nx-3]- 5 *x[nx-4])

    sx = np.ndarray(nx)
    for i in range(2,nx-1):
        sx[i] = (d1*(QQ[i]-QQ[i-1])+d2*(x[i+1]-x[i-1])+d3*(QQ[i+1]-QQ[i-2]))

    sx[0]   = b2* (-23 *QQ0    +21*QQ[0]   +3*QQ[1]    -QQ[2])
    sx[1]   = (d1*(QQ[1]-QQ[0])+ d2 *(x[2]-x[0])+d3*(QQ[2]-QQ0))
    sx[nx-1]=-b2* (-23 *QQ[nx-1] +21*QQ[nx-2]+3*QQ[nx-3] -QQ[nx-4])

    dfdx = fx/sx

    # Exception for derivative at the boundary
    dfdx[0]   =(f[1]-f[0])/(x[1]-x[0])
    dfdx[nx-1]=(f[nx-1]-f[nx-2])/(x[nx-1]-x[nx-2])


    return dfdx
