import numpy as np
import cv2
import pickle
import os
import threading


def relu(x):
    return np.maximum(0.0, x)

def relu_grad(pre_act):
    return (pre_act > 0).astype(np.float32)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def cross_entropy(probs, y_oh, weights=None, l2=1e-4):
    N = probs.shape[0]
    loss = -np.sum(y_oh * np.log(probs + 1e-12)) / N
    if weights:
        loss += 0.5 * l2 * sum(np.sum(w**2) for w in weights)
    return loss

def one_hot(y, C):
    oh = np.zeros((len(y), C), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m, self.v = {}, {}

    def step(self, pid, p, g):
        if pid not in self.m:
            self.m[pid] = np.zeros_like(p)
            self.v[pid] = np.zeros_like(p)
        self.t += 1
        self.m[pid] = self.b1 * self.m[pid] + (1 - self.b1) * g
        self.v[pid] = self.b2 * self.v[pid] + (1 - self.b2) * g**2
        mh = self.m[pid] / (1 - self.b1**self.t)
        vh = self.v[pid] / (1 - self.b2**self.t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps)



class Conv2D:
    def __init__(self, Cin, Cout, k=3, name="c"):
        self.name, self.k = name, k
        scale = np.sqrt(2.0 / (Cin * k * k))
        self.W = (np.random.randn(Cout, Cin, k, k) * scale).astype(np.float32)
        self.b = np.zeros(Cout, dtype=np.float32)
        self.dW = self.db = self._cache = None

    def forward(self, X):
        N, C, H, W = X.shape
        F, _, k, _ = self.W.shape
        p = k // 2
        Xp = np.pad(X, ((0,0),(0,0),(p,p),(p,p)))
        out = np.zeros((N, F, H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                out[:,:,i,j] = np.einsum('nckl,fckl->nf', Xp[:,:,i:i+k,j:j+k], self.W) + self.b
        self._cache = (X, Xp, p)
        return out

    def backward(self, dout):
        X, Xp, p = self._cache
        N, C, H, W = X.shape
        F, _, k, _ = self.W.shape
        self.dW = np.zeros_like(self.W)
        self.db = dout.sum(axis=(0,2,3))
        dXp = np.zeros_like(Xp)
        for i in range(H):
            for j in range(W):
                d = dout[:,:,i,j]
                self.dW += np.einsum('nf,nckl->fckl', d, Xp[:,:,i:i+k,j:j+k])
                dXp[:,:,i:i+k,j:j+k] += np.einsum('nf,fckl->nckl', d, self.W)
        return dXp[:,:,p:-p,p:-p] if p else dXp


class MaxPool2D:
    def __init__(self):
        self._cache = None

    def forward(self, X):
        N, C, H, W = X.shape
        Ho, Wo = H//2, W//2
        out  = np.zeros((N, C, Ho, Wo), dtype=np.float32)
        mask = np.zeros_like(X, dtype=bool)
        for i in range(Ho):
            for j in range(Wo):
                patch = X[:,:,i*2:(i+1)*2,j*2:(j+1)*2]
                out[:,:,i,j] = patch.max(axis=(2,3))
                mx = out[:,:,i,j][:,:,None,None]
                mask[:,:,i*2:(i+1)*2,j*2:(j+1)*2] |= (patch == mx)
        self._cache = (X.shape, mask)
        return out

    def backward(self, dout):
        shape, mask = self._cache
        dX = np.zeros(shape, dtype=np.float32)
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                d = dout[:,:,i,j][:,:,None,None]
                dX[:,:,i*2:(i+1)*2,j*2:(j+1)*2] += d * mask[:,:,i*2:(i+1)*2,j*2:(j+1)*2]
        return dX


class BatchNorm:
    def __init__(self, D, name="bn"):
        self.name = name
        self.gamma = np.ones(D, dtype=np.float32)
        self.beta  = np.zeros(D, dtype=np.float32)
        self.dgamma = self.dbeta = self._cache = None
        self.rm = np.zeros(D, dtype=np.float32)
        self.rv = np.ones(D,  dtype=np.float32)
        self.training = True

    def forward(self, X):
        conv = X.ndim == 4
        if conv:
            N,C,H,W = X.shape
            x2 = X.transpose(0,2,3,1).reshape(-1, C)
        else:
            x2 = X
        if self.training:
            mu  = x2.mean(0); var = x2.var(0)
            xh  = (x2 - mu) / np.sqrt(var + 1e-5)
            self.rm = 0.9*self.rm + 0.1*mu
            self.rv = 0.9*self.rv + 0.1*var
            self._cache = (x2, xh, mu, var, conv, X.shape if conv else None)
        else:
            xh = (x2 - self.rm) / np.sqrt(self.rv + 1e-5)
        out = self.gamma * xh + self.beta
        if conv:
            N,C,H,W = X.shape
            return out.reshape(N,H,W,C).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        x2, xh, mu, var, conv, shape4d = self._cache
        if conv:
            N,C,H,W = shape4d
            dout = dout.transpose(0,2,3,1).reshape(-1, C)
        N = dout.shape[0]
        self.dgamma = (dout * xh).sum(0)
        self.dbeta  = dout.sum(0)
        dxh = dout * self.gamma
        si  = 1.0 / np.sqrt(var + 1e-5)
        dx  = si / N * (N*dxh - dxh.sum(0) - xh*(dxh*xh).sum(0))
        if conv:
            N,C,H,W = shape4d
            dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)
        return dx


class Dense:
    def __init__(self, Din, Dout, name="fc"):
        self.name = name
        self.W = (np.random.randn(Din, Dout) * np.sqrt(2.0/Din)).astype(np.float32)
        self.b = np.zeros(Dout, dtype=np.float32)
        self.dW = self.db = self._X = None

    def forward(self, X):
        self._X = X
        return X @ self.W + self.b

    def backward(self, dout):
        self.dW = self._X.T @ dout
        self.db = dout.sum(0)
        return dout @ self.W.T


class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate; self.mask = None; self.training = True

    def forward(self, X):
        if not self.training: return X
        self.mask = (np.random.rand(*X.shape) > self.rate).astype(np.float32)
        return X * self.mask / (1 - self.rate)

    def backward(self, dout):
        return dout * self.mask / (1 - self.rate) if self.training else dout



def augment(X):
    X = X.copy()
    flip = np.random.rand(len(X)) > 0.5
    X[flip] = X[flip, :, :, ::-1]
    X = np.clip(X * (1 + np.random.uniform(-0.15, 0.15, (len(X),1,1,1))), 0, 1)
    X = np.clip(X + np.random.randn(*X.shape).astype(np.float32) * 0.015, 0, 1)
    return X


IMG_SIZE = 48

class FaceRecognitionCNN:
    """
    Custom 3-block CNN. No pretrained weights. Pure NumPy.
    Input : (N, 1, 48, 48) grayscale float32 in [0,1]
    Output: (N, num_classes) softmax probabilities
    """

    def __init__(self, num_classes, l2=1e-4):
        self.num_classes = num_classes
        self.l2          = l2
        self.label_names = []

        self.conv1 = Conv2D(1,   32, 3, name="c1")
        self.bn1   = BatchNorm(32,    name="bn1")
        self.pool1 = MaxPool2D()

        self.conv2 = Conv2D(32, 64, 3, name="c2")
        self.bn2   = BatchNorm(64,    name="bn2")
        self.pool2 = MaxPool2D()

        self.conv3 = Conv2D(64, 128, 3, name="c3")
        self.bn3   = BatchNorm(128,   name="bn3")
        self.pool3 = MaxPool2D()

        flat_dim   = 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)
        self.fc1   = Dense(flat_dim, 256, name="fc1")
        self.bn4   = BatchNorm(256,       name="bn4")
        self.drop  = Dropout(0.5)
        self.fc2   = Dense(256, num_classes, name="fc2")

    def _mode(self, training):
        for bn in [self.bn1, self.bn2, self.bn3, self.bn4]:
            bn.training = training
        self.drop.training = training

    def forward(self, X):
        x = relu(self.bn1.forward(self.conv1.forward(X)));  x = self.pool1.forward(x)
        x = relu(self.bn2.forward(self.conv2.forward(x)));  x = self.pool2.forward(x)
        x = relu(self.bn3.forward(self.conv3.forward(x)));  x = self.pool3.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = relu(self.bn4.forward(self.fc1.forward(x)))
        x = self.drop.forward(x)
        return softmax(self.fc2.forward(x))

    def backward(self, probs, y_oh):
        N = probs.shape[0]
        d = (probs - y_oh) / N
        d = self.fc2.backward(d)
        d = self.drop.backward(d)
        d = self.bn4.backward(d)
        pre = self.fc1._X @ self.fc1.W + self.fc1.b
        d = d * relu_grad(pre)
        d = self.fc1.backward(d)
        d = d.reshape(N, 128, IMG_SIZE//8, IMG_SIZE//8)
        d = self.pool3.backward(d); d = self.conv3.backward(self.bn3.backward(d))
        d = self.pool2.backward(d); d = self.conv2.backward(self.bn2.backward(d))
        d = self.pool1.backward(d); self.conv1.backward(self.bn1.backward(d))

    def _update(self, adam):
        pairs = [
            ("c1W",self.conv1,"W"),("c1b",self.conv1,"b"),
            ("c2W",self.conv2,"W"),("c2b",self.conv2,"b"),
            ("c3W",self.conv3,"W"),("c3b",self.conv3,"b"),
            ("b1g",self.bn1,"gamma"),("b1b",self.bn1,"beta"),
            ("b2g",self.bn2,"gamma"),("b2b",self.bn2,"beta"),
            ("b3g",self.bn3,"gamma"),("b3b",self.bn3,"beta"),
            ("b4g",self.bn4,"gamma"),("b4b",self.bn4,"beta"),
            ("f1W",self.fc1,"W"),("f1b",self.fc1,"b"),
            ("f2W",self.fc2,"W"),("f2b",self.fc2,"b"),
        ]
        grads = {
            "c1W":self.conv1.dW,"c1b":self.conv1.db,
            "c2W":self.conv2.dW,"c2b":self.conv2.db,
            "c3W":self.conv3.dW,"c3b":self.conv3.db,
            "b1g":self.bn1.dgamma,"b1b":self.bn1.dbeta,
            "b2g":self.bn2.dgamma,"b2b":self.bn2.dbeta,
            "b3g":self.bn3.dgamma,"b3b":self.bn3.dbeta,
            "b4g":self.bn4.dgamma,"b4b":self.bn4.dbeta,
            "f1W":self.fc1.dW,"f1b":self.fc1.db,
            "f2W":self.fc2.dW,"f2b":self.fc2.db,
        }
        for pid, layer, attr in pairs:
            g = grads.get(pid)
            if g is None: continue
            p = getattr(layer, attr)
            if attr == "W": g = g + self.l2 * p
            setattr(layer, attr, adam.step(pid, p, g))

    def train(self, X, y, epochs=30, lr=0.001, batch=16):
        adam = Adam(lr)
        N    = len(X)

        unique, counts = np.unique(y, return_counts=True)
        class_weights = N / (len(unique) * counts)
        class_weights = class_weights / class_weights.mean()  
        
        print(f"\n{'='*52}")
        print(f"  Training Custom CNN from scratch")
        print(f"  Classes : {self.label_names}")
        print(f"  Samples : {N} | Epochs: {epochs} | LR: {lr}")
        print(f"  Class weights: {class_weights}")
        print(f"{'='*52}")
        for ep in range(1, epochs+1):
            self._mode(True)
            idx = np.random.permutation(N)
            Xe, ye = X[idx], y[idx]
            total_loss, correct = 0.0, 0
            nb = max(1, N // batch)
            for b in range(0, N, batch):
                Xb  = augment(Xe[b:b+batch])
                yb  = ye[b:b+batch]
                yoh = one_hot(yb, self.num_classes)
                p   = self.forward(Xb)
                # Apply class weights to loss
                weighted_yoh = yoh * class_weights[yb][:, np.newaxis]
                total_loss += cross_entropy(p, weighted_yoh,
                    weights=[self.conv1.W, self.conv2.W, self.conv3.W, self.fc1.W, self.fc2.W],
                    l2=self.l2)
                correct += (p.argmax(1) == yb).sum()
                self.backward(p, weighted_yoh)
                self._update(adam)
            acc = correct / N * 100
            bar = "█" * int(acc/5) + "░" * (20 - int(acc/5))
            print(f"  Ep {ep:3d}/{epochs}  loss={total_loss/nb:.4f}  acc={acc:5.1f}%  [{bar}]")
        print(f"{'='*52}\n  Done!\n")

    def predict_single(self, face_arr, confidence_threshold=0.5):
        """face_arr: (1,48,48) from face_to_tensor(). Adds batch dim before forward.
        Returns (idx, confidence, name) or (idx, confidence, 'Unknown') if below threshold."""
        self._mode(False)
        X     = face_arr[np.newaxis]   # (1,1,48,48)
        probs = self.forward(X)[0]
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
        # Return 'Unknown' if confidence is too low
        name = self.label_names[idx] if conf >= confidence_threshold else "Unknown"
        return idx, conf, name

    def save(self, path="face_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"  [Saved] {path}  ({os.path.getsize(path)//1024} KB)")

    @classmethod
    def load(cls, path="face_model.pkl"):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"  [Loaded] classes: {obj.label_names}")
        return obj


def build_detector():
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(xml)
    if det.empty():
        raise RuntimeError("Haar cascade XML not found. Reinstall opencv-python.")
    return det

def detect_face(gray, det):
    faces = det.detectMultiScale(gray, 1.3, 5, minSize=(60,60))
    if len(faces) == 0:
        return None
    return sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

def face_to_tensor(gray, box):
    """Returns (1, 48, 48) float32 — one grayscale face crop."""
    x, y, w, h = box
    roi = cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
    return (roi.astype(np.float32) / 255.0)[np.newaxis]   


FONT    = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN = (50, 220, 90)
C_RED   = (50, 60, 230)
C_YEL   = (30, 200, 240)
C_WHITE = (240, 240, 240)
C_BLACK = (0, 0, 0)
C_DARK  = (15, 15, 15)

def txt(img, text, pos, scale=0.6, color=C_WHITE, thick=1):
    cv2.putText(img, text, pos, FONT, scale, C_BLACK, thick+2)
    cv2.putText(img, text, pos, FONT, scale, color,   thick)

def face_box(frame, box, color, label="", conf=None):
    x, y, w, h = box
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
    L = 18
    for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*L,py),color,3)
        cv2.line(frame,(px,py),(px,py+dy*L),color,3)
    if label:
        lbl = f"{label} {conf*100:.0f}%" if conf is not None else label
        (tw,th),_ = cv2.getTextSize(lbl, FONT, 0.65, 1)
        cv2.rectangle(frame,(x,y-th-12),(x+tw+10,y),color,-1)
        cv2.putText(frame, lbl,(x+5,y-5),FONT,0.65,C_BLACK,2)

def hud(frame, lines, y0=26, color=C_WHITE):
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(frame.shape[1], y0+len(lines)*28+6),(0,0,0),-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
    for i,line in enumerate(lines):
        txt(frame, line, (12, y0+i*26), color=color)

def status_bar(frame, msg, color=C_WHITE):
    H = frame.shape[0]
    ov = frame.copy()
    cv2.rectangle(ov,(0,H-28),(frame.shape[1],H),(0,0,0),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    txt(frame, msg, (10,H-8), scale=0.5, color=color)

def progress_bar(frame, n, total):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,H-22),(W,H),C_DARK,-1)
    cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
    filled = int(W * n / total)
    cv2.rectangle(frame,(0,H-22),(filled,H),C_GREEN,-1)
    txt(frame,f"Captured: {n}/{total}",(W//2-60,H-6),scale=0.5)



def show_menu(dataset, model):
    enrolled = list(dataset.keys())
    model_info = f"ready ({len(model.label_names)} people)" if model else "not trained"

    print("\n" + "─"*52)
    print("  FACE RECOGNITION ATTENDANCE SYSTEM")
    print("─"*52)
    print(f"  Enrolled  : {', '.join(enrolled) if enrolled else 'nobody'}")
    print(f"  Model     : {model_info}")
    print("─"*52)
    print("  1. Enroll a new person")
    print("  2. Train model on enrolled faces  (min 1 person)")
    print("  3. Start live recognition")
    print("  4. Load saved model from disk")
    print("  5. Save current model to disk")
    print("  q. Quit")
    print("─"*52)
    return input("  Choose [1/2/3/4/5/q]: ").strip().lower()


SAMPLES_NEEDED = 40
MODEL_FILE     = "face_model.pkl"

def main():
    print("\n" + "="*52)
    print("  Starting camera…")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    det     = build_detector()
    dataset = {}          
    model   = None        
    WIN = "FaceAttend — Face Recognition"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    mode         = "preview"
    enroll_name  = ""
    enroll_buf   = []
    live_result  = ("", 0.0)
    status_msg   = ("Camera ready. Switch to terminal and choose an option.", C_WHITE)

    def set_status(msg, color=C_WHITE):
        nonlocal status_msg
        status_msg = (msg, color)

    def get_user_action():
        return show_menu(dataset, model)

    running = True

    while running:
        if mode == "preview":
            choice = get_user_action()

            if choice == "1":
                print("  Enter person name (first and last name, e.g. John Doe): ", end="", flush=True)
                name = input().strip()
                if not name:
                    print("  [!] Name cannot be empty.")
                    continue
                if "  " in name:
                    name = " ".join(name.split())   
                    print(f"  [!] Extra spaces removed. Using name: '{name}'")
                enroll_name = name
                enroll_buf  = []
                mode        = "enroll"
                set_status(f"ENROLL: '{name}' — look at camera, press SPACE in webcam window to capture", C_YEL)
                print(f"\n  [Enroll] '{name}' — webcam window is open.")
                print(f"  Click on the webcam window, then press SPACE to capture samples.")
                print(f"  Need {SAMPLES_NEEDED} captures. Press ESC to cancel.\n")

                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    box   = detect_face(gray, det)

                    if box is not None:
                        face_box(frame, box, C_YEL, label=f"Enrolling: {enroll_name}")

                    hud(frame, [
                        f"ENROLL MODE — {enroll_name}",
                        "Move your head slightly between captures",
                        "SPACE = capture  |  ESC = cancel",
                    ], color=C_YEL)
                    progress_bar(frame, len(enroll_buf), SAMPLES_NEEDED)
                    status_bar(frame, status_msg[0], status_msg[1])
                    cv2.imshow(WIN, frame)

                    key = cv2.waitKey(30) & 0xFF

                    if key == ord(' '):  
                        if box is not None:
                            enroll_buf.append(face_to_tensor(gray, box))
                            n = len(enroll_buf)
                            set_status(f"Captured {n}/{SAMPLES_NEEDED} for '{enroll_name}'", C_YEL)
                            print(f"  Captured {n}/{SAMPLES_NEEDED}")
                            if n >= SAMPLES_NEEDED:
                                dataset[enroll_name] = list(enroll_buf)
                                set_status(f"✓ '{enroll_name}' enrolled! Choose next action in terminal.", C_GREEN)
                                print(f"\n  [✓] '{enroll_name}' enrolled with {n} samples.")
                                break
                        else:
                            set_status("No face detected — move into frame", C_RED)
                            print("  No face detected.")

                    elif key == 27:     
                        set_status("Enrolment cancelled.", C_WHITE)
                        print("  [Cancelled] Enrolment cancelled.")
                        break

                mode = "preview"

            elif choice == "2":
                if len(dataset) < 1:
                    print("  [!] Enroll at least 1 person before training.")
                    continue

                names = sorted(dataset.keys())
                X_list, y_list = [], []
                for i, name in enumerate(names):
                    for t in dataset[name]:
                        X_list.append(t)   
                        y_list.append(i)

                if len(names) == 1:
                    print("  [Info] Only 1 person enrolled.")
                    print("  [Info] Adding synthetic 'Unknown' class so model can train.")
                    known_samples = X_list[0].shape  
                    n_unknown = max(20, len(X_list))
                    for _ in range(n_unknown):
                        noise = np.random.rand(1, IMG_SIZE, IMG_SIZE).astype(np.float32)
                        X_list.append(noise)
                        y_list.append(1)   
                    names = names + ["Unknown"]
                    print(f"  [Info] Training with classes: {names}")

                X = np.stack(X_list) 
                y = np.array(y_list, dtype=np.int32)

                model = FaceRecognitionCNN(num_classes=len(names))
                model.label_names = names
                model.train(X, y, epochs=30, lr=0.001, batch=16)
                set_status(f"Model trained on {names}. Choose 3 to recognise.", C_GREEN)

            elif choice == "3":
                if model is None:
                    print("  [!] No model available. Train (2) or Load (4) first.")
                    continue

                mode = "recognize"
                set_status("RECOGNITION ACTIVE — press ESC in webcam window to stop", C_GREEN)
                print(f"\n  [Recognition] Running. Press ESC in the webcam window to stop.\n")

                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    box   = detect_face(gray, det)

                    if box is not None:
                        tensor = face_to_tensor(gray, box)
                        _, conf, name = model.predict_single(tensor)
                        live_result   = (name, conf)
                        color = C_GREEN if conf > 0.60 else C_RED
                        face_box(frame, box, color, label=name, conf=conf)

                    name_r, conf_r = live_result
                    hud(frame, [
                        "RECOGNITION MODE",
                        f"Enrolled: {', '.join(model.label_names)}",
                        "ESC = stop",
                    ], color=C_GREEN)
                    if name_r:
                        conf_color = C_GREEN if conf_r > 0.60 else C_RED
                        txt(frame, f"→ {name_r}  ({conf_r*100:.0f}%)",
                            (10, frame.shape[0]//2), scale=1.2, color=conf_color, thick=2)
                    status_bar(frame, status_msg[0], status_msg[1])
                    cv2.imshow(WIN, frame)

                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:    # ESC
                        live_result = ("", 0.0)
                        set_status("Recognition stopped. Choose next action in terminal.", C_WHITE)
                        print("  [Stopped] Recognition stopped.")
                        break

                mode = "preview"

            elif choice == "4":
                print(f"  Enter model file path (press Enter for default '{MODEL_FILE}'): ", end="", flush=True)
                raw = input().strip()
                path = raw if raw else MODEL_FILE
                print(f"  Looking for: {os.path.abspath(path)}")
                if os.path.exists(path):
                    model = FaceRecognitionCNN.load(path)
                    set_status(f"Model loaded: {model.label_names}", C_GREEN)
                else:
                    print(f"  [!] File not found: {os.path.abspath(path)}")
                    print(f"  [Tip] Train first (option 2) — it auto-saves to {MODEL_FILE}")

            elif choice == "5":
                if model is None:
                    print("  [!] No model to save. Train first (option 2).")
                else:
                    print(f"  Save to (press Enter for default '{MODEL_FILE}'): ", end="", flush=True)
                    raw = input().strip()
                    path = raw if raw else MODEL_FILE
                    model.save(path)

            elif choice == "q":
                print("  [Exit] Bye!")
                running = False
                break

            else:
                print("  [!] Invalid choice. Enter 1/2/3/4/5/q")

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        box   = detect_face(gray, det)
        if box is not None:
            x,y,w,h = box
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),1)
        hud(frame,["PREVIEW — switch to terminal to choose an action"], color=C_WHITE)
        status_bar(frame, status_msg[0], status_msg[1])
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()