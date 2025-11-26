from update_vectorstore import VectorstoreUpdater

updater = VectorstoreUpdater()
for kw in ['X-100','x-100','Y-200','G-300','Z-500','音頻設備']:
    results = updater.search_products(kw, limit=10)
    print('KW:', kw, '->', len(results), 'matches')
    for r in results:
        print('  -', r['product_name'], r['category'], r.get('price'))
    print()
