# Quick Deploy to Render (5 Minutes)

## **Step-by-Step Deployment**

### **1. Push Code to GitHub**

```bash
cd e:\ImageVideoDFD

# Initialize Git if not done
git init
git add .
git commit -m "Initial deepfake detection app"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
git branch -M main
git push -u origin main
```

### **2. Create Render Account**

1. Go to https://render.com
2. Click "Sign up"
3. Select "GitHub" and authorize
4. Click "Connect GitHub account"

### **3. Create Web Service**

1. In Render dashboard, click **"New +"**
2. Select **"Web Service"**
3. Select your **`deepfake-detection`** repository
4. Fill in settings:

```
Name: deepfake-detector
Runtime: Python 3
Region: US (Oregon) or your closest region
Branch: main
Root Directory: (leave blank)

Build Command:
pip install -r requirements.txt requirements-prod.txt

Start Command:
gunicorn --workers 2 --worker-class sync --bind 0.0.0.0:$PORT --timeout 120 server:app
```

5. Scroll to "Environment Variables" section
6. Click "Add Environment Variable"
7. Add these variables:

```
SECRET_KEY = (generate random: python -c "import secrets; print(secrets.token_hex(32))")
FLASK_ENV = production
SQLALCHEMY_DATABASE_URI = sqlite:////opt/render/project/src/instance/users.db
PORT = 10000
```

### **4. Deploy**

1. Click **"Create Web Service"**
2. Wait 3-5 minutes for deployment
3. View logs in real-time
4. Once complete, you'll see: ✅ **Live** with a URL like:
   ```
   https://deepfake-detector.onrender.com
   ```

### **5. Test Your Deployment**

```bash
# Test health endpoint
curl https://deepfake-detector.onrender.com/health

# Visit in browser
https://deepfake-detector.onrender.com
```

---

## **Enable Auto-Deploy**

Render auto-deploys on each push to GitHub!

```bash
# Make a change
echo "# Updated" >> README.md

# Push
git add README.md
git commit -m "Update README"
git push origin main

# Render auto-deploys within seconds
```

---

## **Common Issues & Fixes**

### **Issue: Deployment fails with model loading error**
**Fix:** Models need to be in the repo
```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

### **Issue: "Out of memory" during build**
**Fix:** Use Render's Build Cache
- In Render dashboard → Settings → Check "Enable Build Cache"
- Redeploy

### **Issue: Database not persisting**
**Fix:** Use PostgreSQL instead (Render has free tier)
```
SQLALCHEMY_DATABASE_URI = postgresql://user:password@db.render.internal:5432/dbname
```

### **Issue: Models not loading in production**
**Fix:** Check file permissions
```bash
chmod 644 models/*.h5 models/*.pt
git add -A
git commit -m "Fix model permissions"
git push
```

---

## **Upgrade to Paid Plan (Optional)**

Current: Free tier (auto-sleeps after 15 min)

To prevent auto-sleep:

1. In Render dashboard → Select service
2. Click "Settings"
3. Scroll to "Plan"
4. Select **"Starter Plus"** ($7/month)
5. Click "Upgrade"

This gives you:
- ✅ Always running (no auto-sleep)
- ✅ More resources
- ✅ Better performance
- ✅ Email support

---

## **Monitor Your App**

### **In Render Dashboard:**
- View logs in real-time
- See deployment history
- Monitor performance metrics
- Manage environment variables
- Scale instances

### **View Logs:**
```bash
# Live logs (in Render dashboard)
Or via CLI:
render logs --service-id your-service-id
```

---

## **Custom Domain (Optional)**

1. In Render dashboard → Select service
2. Click "Settings"
3. Scroll to "Custom Domain"
4. Enter: `deepfake-detector.yourdomain.com`
5. Update DNS records at your registrar with CNAME

---

## **Troubleshooting Deployment**

### **Build fails:**
```
Check build logs in Render dashboard for specific error
Usually: missing dependencies or model file issues
```

### **App crashes after deploy:**
```
1. Check "Logs" tab in Render dashboard
2. Common causes:
   - Missing environment variable
   - Model file not found
   - Insufficient memory
3. Click "Redeploy" to retry
```

### **Upload not working:**
```
Render uses ephemeral storage - files delete after redeploy
Solution: Configure cloud storage (AWS S3)
```

---

## **Full Render Deployment Architecture**

```
Your GitHub Repo
      ↓
   (push)
      ↓
Render Webhook triggered
      ↓
Build Docker image
      ↓
Run tests (optional)
      ↓
Deploy to live server
      ↓
HTTPS/SSL auto-enabled
      ↓
Your app is live!
```

---

## **Next Steps**

✅ **Deployed successfully?**

1. Share your URL: `https://deepfake-detector.onrender.com`
2. Test with real deepfakes
3. Monitor performance
4. Gather feedback
5. Scale if needed

---

## **Pro Tips**

- **GitHub Integration**: Auto-deploys every push
- **Environment Variables**: Secure - never in code
- **Scaling**: Render scales automatically
- **SSL**: Free HTTPS certificate included
- **Monitoring**: Check logs daily first week

---

**Your app is now live on the internet! 🚀**
