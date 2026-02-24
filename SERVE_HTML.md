# 📄 How to Serve INDEX.html

The `INDEX.html` file is a professional interactive dashboard. Follow one of these methods to view it:

---

## **Method 1: Quick Bash Script (Recommended) ⚡**

```bash
# Make it executable
chmod +x serve-html.sh

# Run it (starts server on port 8080)
./serve-html.sh

# Or specify a different port
./serve-html.sh 3000
```

Then open: **http://localhost:8080**

---

## **Method 2: Python (No Script) 🐍**

```bash
# Python 3
python3 -m http.server 8080

# Python 2
python -m SimpleHTTPServer 8080
```

Then open: **http://localhost:8080**

---

## **Method 3: Node.js (if you have it) 📦**

```bash
# Install http-server globally
npm install -g http-server

# Serve current directory
http-server .

# Or specify port
http-server . -p 8080
```

---

## **Method 4: Docker 🐳**

```bash
docker run -p 8080:80 -v $(pwd):/usr/share/nginx/html nginx
```

Then open: **http://localhost:8080**

---

## **Method 5: Direct Browser (Limited) 🌐**

You can also open the file directly in your browser:
- **macOS:** `open INDEX.html`
- **Linux:** `xdg-open INDEX.html`
- **Windows:** `start INDEX.html`

⚠️ Note: Direct file:// access may have some limitations with relative paths.

---

## **What You'll See**

Once served, you'll see a professional dashboard with:

✅ **Header** — Framework overview
✅ **Validation Results** — 3 real cases with results
✅ **7 Layers of Confidence** — Complete summary table
✅ **4 Phases Delivered** — All project components
✅ **Framework Metrics** — Tests, proxies, cases
✅ **Credibility Transformation** — Before/After comparison
✅ **Quick Navigation** — By role (visitor/practitioner/researcher/auditor)
✅ **Call-to-Action** — How to get started
✅ **Footer** — Links and status

---

## **Recommended Path**

1. **Open INDEX.html** (this dashboard)
2. **Read INDEX.md** (comprehensive overview)
3. **Read README.md** (quick start)
4. **Read PROJECT_STRUCTURE.md** (complete course)
5. **Run the tests** (26+ tests, all passing)
6. **Deploy in production** (finance, AI, infrastructure)

---

## **Troubleshooting**

**Problem:** "Server is not responding"
- **Solution:** Check that the port is not already in use. Try a different port.

**Problem:** "CSS not loading, page looks broken"
- **Solution:** CSS is embedded in the HTML. Try clearing browser cache (Ctrl+F5 or Cmd+Shift+R)

**Problem:** "Links don't work"
- **Solution:** Links are intentionally text-based (not href) because this is a standalone dashboard. Refer to the files directly in your repository.

---

## **Next Steps**

Once you have the HTML loaded:

```bash
# Run all tests
pytest tests/ -v

# Run an audit
python3 tools/run_alignment_audit.py --dataset your_data.csv --sector finance

# View real cases
cat reports/real_cases/case_001/report.md
cat reports/real_cases/case_002/report.md
cat reports/real_cases/case_003/report.md
```

---

**Status:** ✅ Production-Ready (v1.0.0, 2026-02-24)
