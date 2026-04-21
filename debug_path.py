from pathlib import Path
import sys
sys.path.append('src')

# Simuler le contexte de l'API
api_file = Path('src/api.py')
page = api_file.resolve().parents[1] / "web" / "dashboard.html"

print('Chemin calculé:', page)
print('Existe:', page.exists())
print('Chemin absolu:', page.absolute())

# Tester directement
if page.exists():
    print('✅ Fichier trouvé')
else:
    print('❌ Fichier non trouvé')