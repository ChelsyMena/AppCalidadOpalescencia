from waitress import serve
from app_prod import app

serve(app.server, host='0.0.0.0', port=8050)
