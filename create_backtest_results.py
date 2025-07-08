#!/usr/bin/env python
"""
Script to create backtest results for testing the history page
"""

import os
import sys
import django
import json
import requests

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Strategy, DataFile, BacktestResult

def create_backtest_results():
    """Create some backtest results in the database"""
    
    # Check if we have strategies and data files
    strategies = Strategy.objects.all()
    datafiles = DataFile.objects.all()
    
    print(f"Found {strategies.count()} strategies and {datafiles.count()} data files")
    
    if strategies.count() == 0 or datafiles.count() == 0:
        print("No strategies or data files found. Please upload some first.")
        return
    
    # Get the first strategy and data file
    strategy = strategies.first()
    datafile = datafiles.first()
    
    print(f"Using strategy: {strategy.name}")
    print(f"Using data file: {datafile.name}")
    
    # Create a sample backtest result
    sample_result = {
        'summary': {
            'total_trades': 15,
            'total_profit': 1250.50,
            'winning_trades': 9,
            'losing_trades': 6,
            'win_rate': 60.0,
            'avg_profit': 83.37,
            'total_return': 5.2,
            'initial_price': 24000.0,
            'final_price': 25248.0
        },
        'trades': [
            {
                'date': '2024-06-20',
                'type': 'BUY',
                'price': 23567.0,
                'shares': 100,
                'profit': None
            },
            {
                'date': '2024-06-25',
                'type': 'SELL',
                'price': 23721.3,
                'shares': 100,
                'profit': 154.3
            },
            {
                'date': '2024-06-26',
                'type': 'BUY',
                'price': 23868.8,
                'shares': 100,
                'profit': None
            },
            {
                'date': '2024-06-28',
                'type': 'SELL',
                'price': 24010.6,
                'shares': 100,
                'profit': 141.8
            },
            {
                'date': '2024-07-01',
                'type': 'BUY',
                'price': 24141.95,
                'shares': 100,
                'profit': None
            },
            {
                'date': '2024-07-05',
                'type': 'SELL',
                'price': 24323.85,
                'shares': 100,
                'profit': 181.9
            }
        ],
        'currency': 'INR'
    }
    
    # Create multiple backtest results
    for i in range(3):
        backtest_result = BacktestResult.objects.create(
            strategy=strategy,
            datafile=datafile,
            result=sample_result
        )
        print(f"Created backtest result {i+1} with ID: {backtest_result.id}")
    
    print(f"Total backtest results in database: {BacktestResult.objects.count()}")

def run_backtest_via_api():
    """Run a backtest via the API to create real results"""
    
    # Get the first strategy and data file
    strategies = Strategy.objects.all()
    datafiles = DataFile.objects.all()
    
    if strategies.count() == 0 or datafiles.count() == 0:
        print("No strategies or data files found. Please upload some first.")
        return
    
    strategy = strategies.first()
    datafile = datafiles.first()
    
    print(f"Running backtest with strategy: {strategy.name}")
    print(f"Using data file: {datafile.name}")
    
    # Make API call to run backtest
    url = "http://127.0.0.1:8000/api/run-backtest/"
    payload = {
        "strategy_id": strategy.id,
        "datafile_id": datafile.id
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Backtest completed successfully! ID: {result.get('id')}")
            print(f"Total backtest results in database: {BacktestResult.objects.count()}")
        else:
            print(f"Backtest failed with status {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to the API. Make sure the Django server is running on port 8000.")
    except Exception as e:
        print(f"Error running backtest: {e}")

if __name__ == "__main__":
    print("Creating backtest results...")
    
    # First try to run a real backtest via API
    print("\n1. Trying to run real backtest via API...")
    run_backtest_via_api()
    
    # If that fails, create sample results
    print("\n2. Creating sample backtest results...")
    create_backtest_results()
    
    print("\nDone! Check your Results tab now.") 