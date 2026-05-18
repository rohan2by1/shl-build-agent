import os
import re
import glob
import requests

# Your running FastAPI endpoint
API_URL = "http://localhost:8000/chat"
TRACES_DIR = "data/traces"

def parse_markdown_trace(filepath):
    """
    Parses a single Markdown trace file to extract user turns and the expected final URLs.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    trace_id = os.path.basename(filepath).replace(".md", "")

    # 1. Extract User Turns
    # This regex finds everything between "**User**" and "**Agent**"
    user_blocks = re.findall(r'\*\*User\*\*\s+(.*?)\s+\*\*Agent\*\*', content, re.DOTALL)
    
    turns = []
    for block in user_blocks:
        # Remove the blockquote "> " markers and join multi-line messages
        lines = [line.lstrip('> ').strip() for line in block.strip().split('\n') if line.strip()]
        clean_text = " ".join(lines).strip()
        turns.append(clean_text)

    # 2. Extract Expected URLs from the FINAL turn only
    # Split the document by Turns to isolate the last one
    turns_sections = re.split(r'### Turn \d+', content)[1:] 
    
    expected_urls = []
    if turns_sections:
        last_turn_content = turns_sections[-1]
        # Extract all URLs formatted as <https://www.shl.com/...> in the final table
        expected_urls = re.findall(r'<(https://www\.shl\.com/[^>]+)>', last_turn_content)

    return {
        "trace_id": trace_id,
        "turns": turns,
        "expected_urls": expected_urls
    }

def load_all_traces(directory):
    """Loads and parses all markdown traces in the given directory."""
    traces = []
    search_pattern = os.path.join(directory, "*.md")
    for filepath in glob.glob(search_pattern):
        parsed_trace = parse_markdown_trace(filepath)
        # Sort traces to ensure they run in order (C1, C2, C3...)
        traces.append(parsed_trace)
        
    # Sort by the number in the trace ID (e.g., C1 -> 1, C10 -> 10)
    traces.sort(key=lambda x: int(re.search(r'\d+', x['trace_id']).group()))
    return traces

def calculate_recall_at_k(expected_urls, recommended_urls, k=10):
    """Calculates Recall@K based on the assignment appendix."""
    if not expected_urls:
        return 0.0
        
    top_k_recommendations = recommended_urls[:k]
    relevant_count = sum(1 for url in expected_urls if url in top_k_recommendations)
    
    return relevant_count / len(expected_urls)

def run_evaluation():
    print(f"Loading traces from {TRACES_DIR}...")
    test_traces = load_all_traces(TRACES_DIR)
    
    if not test_traces:
        print(f"No markdown files found in {TRACES_DIR}. Please check the path.")
        return

    print(f"Successfully loaded {len(test_traces)} traces. Starting evaluation...\n")
    
    total_recall = 0
    completed_traces = 0

    for trace in test_traces:
        print(f"--- Running Trace: {trace['trace_id']} ---")
        conversation_history = []
        final_recommendations = []
        
        for turn_number, user_input in enumerate(trace["turns"], start=1):
            conversation_history.append({"role": "user", "content": user_input})
            
            payload = {"messages": conversation_history}
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                print(f"  [!] Turn {turn_number} FAILED: API Error - {e}")
                break

            agent_reply = data.get("reply", "")
            conversation_history.append({"role": "assistant", "content": agent_reply})
            
            print(f"  Turn {turn_number} User: {user_input[:60]}..." if len(user_input) > 60 else f"  Turn {turn_number} User: {user_input}")
            print(f"  Turn {turn_number} Agent: {agent_reply[:60]}..." if len(agent_reply) > 60 else f"  Turn {turn_number} Agent: {agent_reply}")
            
            # Check if agent correctly signals the end of the conversation
            if data.get("end_of_conversation"):
                print(f"  -> Agent triggered end_of_conversation.")
                final_recommendations = [rec.get("url") for rec in data.get("recommendations", [])]
                break
                
        # Evaluate Recall for this trace
        if not final_recommendations:
            print(f"  [WARNING] Trace {trace['trace_id']} ended without providing recommendations.")
            recall = 0.0
        else:
            recall = calculate_recall_at_k(trace["expected_urls"], final_recommendations, k=10)
            
        print(f"  => Recall@10: {recall:.2f}\n")
        total_recall += recall
        completed_traces += 1

    # Calculate Mean Recall
    if completed_traces > 0:
        mean_recall = total_recall / completed_traces
        print(f"====================================")
        print(f"FINAL MEAN RECALL@10: {mean_recall:.2f}")
        print(f"====================================")

if __name__ == "__main__":
    run_evaluation()