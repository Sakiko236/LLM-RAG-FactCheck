import sqlite3
import os

def get_evidence_text(evidence_id, db_path='../../data/evidence.db'):
    if not os.path.exists(db_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_db_path = os.path.join(script_dir, 'evidence.db')
        if os.path.exists(potential_db_path):
            db_path = potential_db_path
        else:
            print(f"Error: Database file not found at {db_path}")
            return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT text FROM evidence WHERE id = ?', (evidence_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        else:
            return None
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

if __name__ == "__main__":
    print("Evidence Retrieval Tool (Enter 'q' to quit)")
    while True:
        ev_id = input("\nEnter Evidence ID: ").strip()
        if ev_id.lower() in ['q', 'exit', 'quit']:
            break
        if not ev_id:
            continue
            
        text = get_evidence_text(ev_id)
        if text:
            print(f"Result: {text}")
        else:
            print(f"Evidence ID '{ev_id}' not found.")