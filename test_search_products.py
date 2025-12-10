from update_vectorstore import VectorstoreUpdater

updater = VectorstoreUpdater()
results = updater.search_products('音箱', limit=10)
print('Found', len(results), 'matches')
for r in results:
    print('-', r['product_name'], r['category'], r.get('price'))
