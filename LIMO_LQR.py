
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Rastic Limo LQR code
# LQR code adapted by Carter Berlind to fit Bicyle dynamical model (for AgileX Limo)
# For more info on dynimic model, see: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357
# Adapted from LQR code by original: Addison Sears-Collins https://automaticaddison.com
# For more info on LQR and original LQR code see https://automaticaddison.com/linear-quadratic-regulator-lqr-with-python-code-example/
# Description: Linear Quadratic Regulator for AgileX Limo 

 

class LQR:
    def __init__(self, L=0.2, dt=.1, method="dynamic_solve"):
        """
        This class is an object in memory that contains a linear quadratic regulator 
        to act as a tracking controller for the AgileX Limo robot
        Spec sheet for the Limo can be found here: https://github.com/agilexrobotics/limo-doc/blob/master/Limo%20user%20manual(EN).md

        :param L: wheel base in meters, default to 0.2
        :param dt: Time step s
        :param method: String determining whether the solver uses "direct_solve" which uses the scipy Algebraic Riccati Solver,
            or "dynamic_solve" which uses a dynamic programming method
        """
        self.L = L
        self.dt = dt
        self.method = method

        # input cost matrix
        self.R = np.array([
            [10,0], #linear velocity input cost 
            [0,100]  #steering angle velocity input cost
        ])

        # state cost matrix
        self.Q = np.array([
            [1,0,0,0], #x position error cost
            [0,1,0,0], #y position error cost
            [0,0,10,0], #theta position error cost
            [0,0,0,0]  #linear velocity error 
        ])
        # time step size
        

    def getAB(self, theta):
        """
        Calculates and returns the B matrix in x_t = A @ x_prev + B @ u_prev 
        The A matrix is an instantaneous linearization of the nonlinear system
        5x5 matix ---> number of states 
     
        :param theta: 2D orientation of vehicle
        :param delta: steering angle 
        :return: B matrix ---> 4x2 NumPy array
        """
        A = np.array([
            [1,0,0,np.cos(theta)*self.dt],
            [0,1,0,np.sin(theta)*self.dt],
            [0,0,1,0],
            [0,0,0,1]
            ])
        B = np.array([
            [0,np.cos(theta)*self.dt],
            [0,np.sin(theta)*self.dt],
            [self.dt,0],
            [0,1]
        ])
        return A,B
 
     
    def lqr(self, x, xd,time):
        """
        Discrete-time linear quadratic regulator for a nonlinear system.
    
        :param x: The current state of the system 
            4x1 NumPy Array given the state is [x,y,theta, delta] ---> 
            [meters, meters, radians,radians]
        :param xd: The desired state of the system
            4x1 NumPy Array given the state is [x,y,theta, delta] ---> 
            [meters, meters, radians,radians]
        :param dt: The size of the timestep in seconds -> float
    
        :return: u_star: Optimal action u for the current state 
            2x1 NumPy Array given the control input vector is
            [linear velocity of the car, angular velocity of the car]
            [meters per second, radians per second]
        """
        # We want the system to stabilize at xd (our waypoint)
        x_error = x - xd

        A,B =self.getAB(x[2,0])
        # A = np.zeros((5,5))



        if self.method == "dynamic_solve":
            # Solutions to discrete LQR problems are obtained using the dynamic 
            # programming method.
            # The optimal solution is obtained recursively, starting at the last 
            # timestep and working backwards.
            # You can tune this number
            N =time
    
            # Create a list of N + 1 elements
            P = [None] * (N + 1)
    
            # LQR via Dynamic Programming
            P[N] = self.Q
    
            # For i = N, ..., 1
            for i in range(N, 0, -1):
        
                # Discrete-time Algebraic Riccati equation to calculate the optimal 
                # state cost matrix
                P[i-1] = self.Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                    self.R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
        
            
        
            # For i = 0, ..., N - 1
            
        
            # Calculate the optimal feedback gain K
            K = -np.linalg.pinv(self.R + B.T @ P[1] @ B) @ B.T @ P[1] @ A
        
            u = K @ x_error
                
            # Optimal control input is u_star
            u_star = u
        u_star[1,0] = np.clip(u_star[1,0],-1-x[3,0], 1-x[3,0])
        u_star[0,0] = np.clip(u_star[0,0],np.sin(-.7)*(u_star[1,0]+x[3,0])/self.L,np.sin(.7)*(u_star[1,0]+x[3,0])/self.L)
        return u_star
 

if __name__ == '__main__':
    x = np.array([
        [0],
        [0],
        [.0],
        [0]
    ])
    x1 = np.array([
        [.8],
        [.8],
        [1.57],
        [0]
    ])
    x2 = np.array([
        [1.6],
        [1.6],
        [0],
        [0]
    ])

    xs = [x1,x2]
    lqr = LQR(method="dynamic_solve")
    # lqr = LQR()
    poses = x[:2]
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax2 = fig.add_subplot()
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_xlim([0,100])
    ax.set_ylim([0,6])
    line1, = ax.plot(poses[0],poses[1], 'r-')
    # for _ in range(500):
    for xd in xs:
        count = 0
        for i in range(50,1,-1):
            u = lqr.lqr(x,xd,i)
            A,B=lqr.getAB(x[2,0])
            x = A@x+B@u
            poses = np.hstack((poses,x[:2]))
            line1.set_xdata(poses[0][-5:])
            line1.set_ydata(poses[1][-5:])
            fig.canvas.draw()
            fig.canvas.flush_events()
            print(np.linalg.norm(xd[:2]-x[:2]))
            # if np.linalg.norm(xd[:2]-x[:2])<0.1:
            #     break
            count+=1
    
    
