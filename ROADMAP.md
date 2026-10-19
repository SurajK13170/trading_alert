. Ye 5 terminals start karo (order mein)

# Terminal 1 — Redis (agar band ho)
redis-server

# Terminal 2 — Quant Engine
cd ai_engine && python indian_quant_engine.py

# Terminal 3 — AI Engine  
cd ai_engine && python indian_ai_engine.py

# Terminal 4 — Execution Manager
cd ai_engine && python indian_execution_manager.py

# Terminal 5 — Backend
cd backend_manager && node app.js

# Terminal 6 — Frontend
cd frontend_manager && npm run dev