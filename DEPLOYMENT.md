# Deployment Guide - DeepFake Detection Model

## **Quick Start: Choose Your Platform**

### **Recommended Options (Ranked by Ease)**
1. **Render** - Easiest, good free tier, auto-deploys from GitHub
2. **Railway** - Simple, fast deploys, cheap pricing
3. **Heroku** - Industry standard (note: free tier discontinued)
4. **AWS** - Most powerful, highest cost
5. **DigitalOcean** - Good balance of cost & control
6. **Google Cloud** - Enterprise option
7. **Docker locally** - Full control, self-hosted

---

## **Option 1: Deploy to Render (RECOMMENDED - Easiest)**

### **Prerequisites**
- GitHub account with repository
- Render account (https://render.com)

### **Step 1: Prepare Repository**

```bash
cd e:\ImageVideoDFD
git init
git add .
git commit -m "Initial deepfake detection deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
git push -u origin main
```

### **Step 2: Create Render Account**
1. Go to https://render.com
2. Sign up with GitHub
3. Connect your GitHub account

### **Step 3: Deploy Web Service**
1. Click "New +" → "Web Service"
2. Select your repository: `deepfake-detection`
3. Configure settings:
   ```
   Name: deepfake-detector
   Environment: Python
   Build Command: pip install -r requirements.txt requirements-prod.txt
   Start Command: gunicorn --workers 2 --worker-class sync --bind 0.0.0.0:$PORT --timeout 120 server:app
   ```

### **Step 4: Set Environment Variables**
In Render dashboard → Environment:
```
SECRET_KEY=generate-random-key-here
FLASK_ENV=production
SQLALCHEMY_DATABASE_URI=sqlite:////opt/render/project/src/instance/users.db
PORT=10000
```

### **Step 5: Deploy**
- Click "Deploy"
- Wait 3-5 minutes for build
- Access at: `https://deepfake-detector.onrender.com`

---

## **Option 2: Deploy to Railway (Fast & Cheap)**

### **Step 1: Install Railway CLI**
```bash
npm install -g @railway/cli
# OR
curl -fsSL https://railway.app/install.sh | sh
```

### **Step 2: Login & Initialize**
```bash
railway login
cd e:\ImageVideoDFD
railway init
```

### **Step 3: Configure**
```bash
railway env
# Add:
SECRET_KEY=your-secret-key
FLASK_ENV=production
PORT=8000
```

### **Step 4: Deploy**
```bash
railway up
# Get URL:
railway domain
```

---

## **Option 3: Deploy with Docker (Self-Hosted)**

### **Step 1: Build Docker Image**
```bash
cd e:\ImageVideoDFD
docker build -t deepfake-detector:latest .
```

### **Step 2: Run Container Locally**
```bash
docker run -p 5000:10000 \
  -e SECRET_KEY=your-secret-key \
  -e FLASK_ENV=production \
  -v $(pwd)/Uploaded_Files:/app/Uploaded_Files \
  -v $(pwd)/instance:/app/instance \
  deepfake-detector:latest
```

### **Step 3: Deploy to Cloud**
**AWS Elastic Container Service (ECS):**
```bash
# Create ECR repository
aws ecr create-repository --repository-name deepfake-detector

# Push image
docker tag deepfake-detector:latest ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/deepfake-detector:latest
docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/deepfake-detector:latest

# Deploy via ECS console
```

**DigitalOcean App Platform:**
```bash
doctl apps create --spec app.yaml
```

---

## **Option 4: AWS Deployment (Most Powerful)**

### **Using AWS Elastic Beanstalk:**

```bash
# Install EB CLI
pip install awseb-cli

# Initialize
eb init -p python-3.10 deepfake-detector -r us-east-1

# Create environment
eb create deepfake-production

# Deploy
eb deploy

# Monitor
eb open
```

### **Environment Variables (`.ebextensions/01_env.config`):**
```yaml
option_settings:
  aws:elasticbeanstalk:application:environment:
    FLASK_ENV: production
    SECRET_KEY: your-secret-key
    SQLALCHEMY_DATABASE_URI: sqlite:////var/app/current/instance/users.db
```

---

## **Option 5: DigitalOcean Droplet (Most Control)**

### **Step 1: Create Droplet**
- Create Ubuntu 20.04 Droplet (2GB RAM minimum)
- SSH into droplet: `ssh root@YOUR_IP`

### **Step 2: Install Dependencies**
```bash
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx supervisor
```

### **Step 3: Clone & Setup**
```bash
cd /var/www
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt requirements-prod.txt
```

### **Step 4: Configure Supervisor**
Create `/etc/supervisor/conf.d/deepfake.conf`:
```ini
[program:deepfake]
directory=/var/www/deepfake-detection
command=/var/www/deepfake-detection/venv/bin/gunicorn --workers 2 --bind 0.0.0.0:8000 server:app
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/deepfake.log
```

### **Step 5: Configure Nginx**
Create `/etc/nginx/sites-available/deepfake`:
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static/ {
        alias /var/www/deepfake-detection/static/;
    }
}
```

### **Step 6: Enable & Start**
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo systemctl restart nginx
```

---

## **Performance Optimization for Production**

### **1. Gunicorn Workers**
```bash
# Calculate optimal workers: (2 × CPU_cores) + 1
# 2 CPU cores = 5 workers
gunicorn --workers 5 --worker-class sync --bind 0.0.0.0:8000 server:app
```

### **2. Enable Caching**
Add to server.py:
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/health')
@cache.cached(timeout=60)
def health_check():
    return jsonify({'status': 'healthy'})
```

### **3. Database Optimization**
```python
# Use PostgreSQL instead of SQLite for production
# SQLALCHEMY_DATABASE_URI = "postgresql://user:password@db_host/dbname"
```

### **4. Model Optimization**
- Reduce frame extraction: 15 frames (default) → 10 frames (faster)
- Enable model quantization (convert float32 → int8)
- Use GPU if available (add `CUDA_VISIBLE_DEVICES=0`)

---

## **Monitoring & Logging**

### **Enable Application Monitoring**

```python
# Add to server.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler('logs/deepfake.log', 
                                      maxBytes=10240000, 
                                      backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('DeepFake Detection startup')
```

### **Monitor with Sentry**
```bash
pip install sentry-sdk
```

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()],
    traces_sample_rate=0.1
)
```

---

## **Cost Estimates (Monthly)**

| Platform | Starter | Production |
|----------|---------|------------|
| **Render** | $0 (free) | $7-50 |
| **Railway** | $5 | $20-100 |
| **Heroku** | Discontinued | $50+ |
| **DigitalOcean** | $6 | $20-100 |
| **AWS** | $10 | $50-500+ |
| **Google Cloud** | $10 | $50-500+ |

---

## **Security Checklist**

Before deploying to production:

- [ ] Change `SECRET_KEY` to random string: `python -c "import secrets; print(secrets.token_hex(32))"`
- [ ] Set `DEBUG=False` in production
- [ ] Use HTTPS (all platforms provide free SSL)
- [ ] Implement rate limiting
- [ ] Set up firewall rules
- [ ] Enable database backups
- [ ] Configure CORS properly
- [ ] Use environment variables (never hardcode secrets)
- [ ] Set file upload limits
- [ ] Enable logging & monitoring

---

## **Scaling Strategies**

### **When You Need to Scale:**
1. **Single Server → Multiple Servers**
   - Use load balancer (Nginx, HAProxy)
   - Horizontal scaling

2. **Database Bottleneck**
   - Migrate from SQLite → PostgreSQL
   - Add read replicas
   - Implement caching layer (Redis)

3. **Model Processing Bottleneck**
   - Add GPU support
   - Implement job queue (Celery)
   - Use async processing

4. **Storage Bottleneck**
   - Use S3/Cloud Storage
   - Implement cleanup jobs
   - Stream large files

---

## **Post-Deployment Checklist**

After successful deployment:

```bash
✅ Test health endpoint: /health
✅ Test signup/login flow
✅ Test image upload
✅ Test video upload
✅ Verify database created
✅ Check logs for errors
✅ Monitor performance
✅ Set up backups
✅ Configure SSL certificate
✅ Test error handling
✅ Load test with tools like Apache Bench
```

---

## **Troubleshooting**

### **Models not loading**
```bash
# Check model files exist
ls -lh models/
# Verify permissions
chmod 644 models/*.h5 models/*.pt
```

### **Out of memory**
```bash
# Reduce frames extracted
# Update in server.py:
frames, frame_paths = extract_video_frames(video_path, num_frames=10)  # was 15
```

### **Slow inference**
```bash
# Enable GPU if available
export CUDA_VISIBLE_DEVICES=0
# Add workers
gunicorn --workers 4 server:app
```

### **Database locked**
```bash
# Use PostgreSQL instead of SQLite
# Or implement connection pooling
```

---

## **Recommended Deployment Flow**

```
1. Local Testing
   ↓
2. GitHub Push
   ↓
3. Deploy to Staging (Render Free Tier)
   ↓
4. Test Thoroughly
   ↓
5. Deploy to Production
   ↓
6. Monitor & Optimize
   ↓
7. Scale if Needed
```

---

## **Quick Deploy Commands**

### **Render**
```bash
git push origin main  # Auto-deploys
```

### **Docker Local**
```bash
docker-compose up -d
```

### **AWS Beanstalk**
```bash
eb deploy
```

### **Railway**
```bash
railway up
```

### **DigitalOcean**
```bash
doctl apps create --spec app.yaml
```

---

**Which platform would you like detailed steps for?**
Choose one:
1. ✅ **Render** (easiest)
2. Railway
3. Docker locally
4. AWS
5. DigitalOcean
