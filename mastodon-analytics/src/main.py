#!/usr/bin/env python3
import os
import json
import time
import heapq
from datetime import datetime
from collections import defaultdict
from mpi4py import MPI
import ijson  # More memory-efficient JSON parsing

class MastodonAnalytics:
    def __init__(self, input_file, output_dir=None):
        """
        Initialize Mastodon analytics processor
        
        Args:
            input_file (str): Path to the NDJSON file
            output_dir (str, optional): Directory to save results
        """
        self.input_file = input_file
        self.output_dir = output_dir or '.'
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Data storage
        self.hour_sentiment = defaultdict(float)
        self.user_sentiment = defaultdict(lambda: {'username': '', 'sentiment': 0})

    def _extract_hour(self, timestamp):
        """
        Extract hour from timestamp
        
        Args:
            timestamp (str): ISO format timestamp
        
        Returns:
            str: Formatted hour string
        """
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:00')
        except:
            return None

    def _process_chunk(self, chunk):
        """
        Process a chunk of data
        
        Args:
            chunk (list): List of JSON objects to process
        """
        for item in chunk:
            # Skip malformed entries
            if not item or not isinstance(item, dict):
                continue
            
            # Extract key information
            try:
                hour = self._extract_hour(item.get('created_at', ''))
                sentiment = item.get('sentiment', 0)
                account = item.get('account', {})
                user_id = account.get('id')
                username = account.get('username', '')
            except:
                continue
            
            # Skip entries without critical information
            if not hour or not user_id:
                continue
            
            # Accumulate hour sentiment
            if hour:
                self.hour_sentiment[hour] += sentiment
            
            # Accumulate user sentiment
            if user_id not in self.user_sentiment or self.user_sentiment[user_id]['username'] == '':
                self.user_sentiment[user_id] = {
                    'username': username,
                    'sentiment': sentiment
                }
            else:
                self.user_sentiment[user_id]['sentiment'] += sentiment

    def _split_and_process(self):
        """
        Split file and process chunks in parallel
        """
        # Estimate file size and calculate chunk distribution
        file_size = os.path.getsize(self.input_file)
        chunk_size = file_size // self.size
        
        # Calculate start and end positions for this rank
        start_pos = self.rank * chunk_size
        end_pos = (self.rank + 1) * chunk_size if self.rank < self.size - 1 else None
        
        # Chunk processing with ijson for memory efficiency
        chunk = []
        with open(self.input_file, 'rb') as f:
            # Skip to start position for this rank
            f.seek(start_pos)
            
            # If not first rank, skip first line (might be partial)
            if self.rank > 0:
                f.readline()
            
            parser = ijson.parse(f)
            for prefix, event, value in parser:
                # Process chunk when it reaches a good size
                if len(chunk) >= 10000:
                    self._process_chunk(chunk)
                    chunk.clear()
                
                # Collect items
                if prefix == '' and event == 'start_map':
                    current_item = {}
                elif prefix == '' and event == 'map_key':
                    current_key = value
                elif prefix == '' and event == 'string':
                    if current_key:
                        current_item[current_key] = value
                elif prefix == '' and event == 'end_map':
                    chunk.append(current_item)
                    current_item = {}
                    current_key = None
                
                # Stop if reached end position
                if end_pos and f.tell() >= end_pos:
                    break
        
        # Process any remaining chunk
        if chunk:
            self._process_chunk(chunk)

    def _gather_results(self):
        """
        Gather and consolidate results across all ranks
        
        Returns:
            tuple: Consolidated results for hours and users
        """
        # Gather results from all ranks
        all_hour_sentiments = self.comm.gather(self.hour_sentiment, root=0)
        all_user_sentiments = self.comm.gather(self.user_sentiment, root=0)
        
        if self.rank == 0:
            # Merge hour sentiments
            merged_hours = defaultdict(float)
            for hour_dict in all_hour_sentiments:
                for hour, sentiment in hour_dict.items():
                    merged_hours[hour] += sentiment
            
            # Merge user sentiments
            merged_users = {}
            for user_dict in all_user_sentiments:
                for user_id, data in user_dict.items():
                    if user_id not in merged_users:
                        merged_users[user_id] = data
                    else:
                        merged_users[user_id]['sentiment'] += data['sentiment']
            
            return merged_hours, merged_users
        
        return None, None

    def analyze(self):
        """
        Perform full Mastodon data analysis
        """
        start_time = time.time()
        
        # Process data in chunks
        self._split_and_process()
        
        # Gather and process results
        merged_hours, merged_users = self._gather_results()
        
        # Output results (root process only)
        if self.rank == 0:
            # Top 5 happiest and saddest hours
            happiest_hours = heapq.nlargest(5, merged_hours.items(), key=lambda x: x[1])
            saddest_hours = heapq.nsmallest(5, merged_hours.items(), key=lambda x: x[1])
            
            # Top 5 happiest and saddest users
            happiest_users = heapq.nlargest(5, merged_users.items(), 
                                            key=lambda x: x[1]['sentiment'])
            saddest_users = heapq.nsmallest(5, merged_users.items(), 
                                             key=lambda x: x[1]['sentiment'])
            
            # Output results
            self._output_results(happiest_hours, saddest_hours, 
                                 happiest_users, saddest_users)
            
            # Total runtime
            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f} seconds")
    
    def _output_results(self, happiest_hours, saddest_hours, 
                        happiest_users, saddest_users):
        """
        Write results to output files
        """
        # Happiest Hours
        with open(os.path.join(self.output_dir, 'happiest_hours.txt'), 'w') as f:
            f.write("Top 5 Happiest Hours:\n")
            for hour, sentiment in happiest_hours:
                f.write(f"{hour}: {sentiment:.2f}\n")
        
        # Saddest Hours
        with open(os.path.join(self.output_dir, 'saddest_hours.txt'), 'w') as f:
            f.write("Top 5 Saddest Hours:\n")
            for hour, sentiment in saddest_hours:
                f.write(f"{hour}: {sentiment:.2f}\n")
        
        # Happiest Users
        with open(os.path.join(self.output_dir, 'happiest_users.txt'), 'w') as f:
            f.write("Top 5 Happiest Users:\n")
            for user_id, data in happiest_users:
                f.write(f"{data['username']} (ID: {user_id}): {data['sentiment']:.2f}\n")
        
        # Saddest Users
        with open(os.path.join(self.output_dir, 'saddest_users.txt'), 'w') as f:
            f.write("Top 5 Saddest Users:\n")
            for user_id, data in saddest_users:
                f.write(f"{data['username']} (ID: {user_id}): {data['sentiment']:.2f}\n")

def main():
    """
    Main execution function
    """
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Input file path (replace with appropriate path)
    input_file = '/Users/manishghoshal/Desktop/UniMelb/Semester 1/CCC/testing_ccc/data/mastodon-16m.ndjson'
    output_dir = './mastodon_results'
    
    # Only root process prints initialization
    if rank == 0:
        print(f"Processing Mastodon data with {comm.Get_size()} processes")
    
    # Initialize and run analytics
    analytics = MastodonAnalytics(input_file, output_dir)
    analytics.analyze()

if __name__ == "__main__":
    main()