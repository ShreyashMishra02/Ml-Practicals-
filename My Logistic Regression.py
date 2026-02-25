import numpy as np

X1= np.array([1,2,3,4,5],dtype=float)
X2=np.array([60,65,70,75,85],dtype=float)
y=np.array([0,0,1,1,1],dtype=float)
n=len(y)

X1_mean=X1.mean()
X1_std=X1.std()

X2_mean=X2.mean()
X2_std=X2.std()

X1=(X1-X1_mean)/X1_std
X2=(X2-X2_mean)/X2_std

def sigmoid(z):
    return 1/(1+np.exp(-z)) #converts value to value between (0-1)
def log_loss(y,y_pred):
    y_pred=min(max(y_pred, 1e-15), 1- 1e-15)
    if y==1:
        return -np.log(y_pred)
    else:
        return -np.log(1-y_pred)
    
m1= 0.0
m2= 0.0
c= 0.0
learning_rate=0.01
epochs=100

for epoch in range(1,epochs +1):
    total_loss = 0
    print(f"\n======= EPOCH {epoch} ========")
    print("Sample |  x1  |  x2  | y |  z  | y_pred |  loss  |  m1  |  c")
    print("------------------------------------------------------------")
    
    for i in range(n):
        x1 = X1[i]
        x2 = X2[i]
        y_i= y[i]
        
        z= m1 * x1+m2 * x2+ c
        y_pred = sigmoid(z)
        
        loss = log_loss(y_i, y_pred)
        total_loss += loss
        
        error = y_pred - y_i
        #Gradient Descent :- Enhancing The value of m1, m2 , c 
        m1-=learning_rate*error*x1
        m2-=learning_rate*error*x2
        c-=learning_rate*error
        
        print(f"{i+1:^6} | {x1:>7.3f} | {int(y_i):^1} | "
        f"{z:>7.3f} | {y_pred:>6.3f} | {loss:>6.3f} | "
        f"{m1:>6.3f} | {m2:>6.3f} | {c:>6.3f}")
        #Here ^6 mean it will leave 6 spaces afte printing the value.
        #>7.3f 7 means it will leave 7 spaces and .3f means only 3 point values will be shown
        print("---------------------------------------------------------")
        print(f"Epoch {epoch} Average Cost = {total_loss / n:.4f}")
        
    print("-------------------------------------------------------------")
    print(f"Epoch {epoch} Average Cost = {total_loss / n:.4f}")
    
print("\nFinal Model Parameters:")
print(f"m1 = {m1:.4f}")
print(f"m2 = {m2: .4f}")
print(f"c  = {c:.4f}")
print("\n Final Predictions: ")
print("Sample | y_actual | y_pred | class")
print("----------------------------------")
for i in range(n):
    z= m1*X1[i]+m2*X2[i] + c
    y_pred = sigmoid(z)
    cls = 1 if y_pred >= 0.5 else 0
    print(f"{i+1:^6} | {int(y[i]):^8} | {y_pred:>6.3f} | {cls:^5}")
    
print("\n-----USER INPUT PREDICTION-----")
user_x1 = float(input("Enter Study Hours: "))
user_x2 = float(input("Enter Attendence(%): "))

user_x1_scaled = (user_x1 - X1_mean)/ X1_std 
user_x2_scaled = (user_x2 - X2_mean)/ X2_std 

z_user= m1 * user_x1_scaled + m2 * user_x2_scaled + c
y_pred_user = sigmoid(z_user)

user_class = 1 if y_pred_user >= 0.5 else 0 

print("\nPrediction Result: ")
print(f"Predicted Probability (Pass) = {y_pred_user:.4f}")

if user_class == 1: 
    print("Final Decision: PASS")
else: 
    print("Final Decision: FAIL")
