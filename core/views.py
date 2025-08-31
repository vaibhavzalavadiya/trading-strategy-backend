# core/views.py

from django.http import JsonResponse, HttpResponseBadRequest
import pandas as pd
import json
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .strategy_engine import analyze_strategy
import logging
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .models import Strategy, DataFile, BacktestResult
import os
import inspect
# Add all necessary imports for strategy execution
import numpy as np
from math import floor, ceil, sqrt, log, exp, sin, cos, tan, pi, e as math_e
from datetime import datetime, timedelta
import warnings
from django.conf import settings
import time
import chardet

logger = logging.getLogger(__name__)

def backtest_view(request):
    try:
        # Replace with your actual CSV and logic
        df = pd.read_csv("media/data/NIFTY.csv")

        trades = []
        for i in range(3):
            trades.append({
                "symbol": "NIFTY",
                "entry": f"2024-01-0{i+1}",
                "exit": f"2024-01-0{i+2}",
                "profit": 100 * (i + 1)
            })

        result = {
            "message": "Backtest complete!",
            "total_trades": len(trades),
            "profit": sum(t["profit"] for t in trades),
            "trades": trades,
        }

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def smart_convert_csv(df):
    """
    Smart CSV converter that handles any CSV format and converts to standard format.
    Auto-detects common column name variations and converts them.
    """
    try:
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert all column names to lowercase for easier matching
        df.columns = df.columns.str.lower().str.strip()
        
        # Column mapping for common variations
        column_mappings = {
            # Timestamp/Date variations
            'timestamp': ['timestamp', 'time', 'date', 'datetime', 'date_time', 'date time', 
                         't', 'dt', 'datetime', 'date/time', 'date/time', 'date and time',
                         'timestamp_utc', 'utc_timestamp', 'utc timestamp', 'utc time',
                         'local_time', 'local time', 'local_timestamp', 'local timestamp'],
            
            # Open price variations
            'open': ['open', 'opening', 'opening price', 'open price', 'open_price', 
                    'opening_price', 'o', 'open_', 'opening_', 'first', 'first price',
                    'starting', 'starting price', 'start', 'start price'],
            
            # High price variations
            'high': ['high', 'highest', 'high price', 'highest price', 'high_price', 
                    'highest_price', 'h', 'high_', 'highest_', 'max', 'maximum',
                    'max price', 'maximum price', 'upper', 'upper price'],
            
            # Low price variations
            'low': ['low', 'lowest', 'low price', 'lowest price', 'low_price', 
                   'lowest_price', 'l', 'low_', 'lowest_', 'min', 'minimum',
                   'min price', 'minimum price', 'lower', 'lower price'],
            
            # Close price variations
            'close': ['close', 'closing', 'close price', 'closing price', 'close_price', 
                     'closing_price', 'c', 'close_', 'closing_', 'last', 'last price',
                     'final', 'final price', 'end', 'end price'],
            
            # Volume variations
            'volume': ['volume', 'vol', 'amount', 'quantity', 'qty', 'size', 'trade size',
                      'trade_volume', 'trade volume', 'trading volume', 'trading_volume',
                      'contracts', 'contract size', 'contract_size', 'units', 'shares',
                      'shares traded', 'shares_traded', 'traded shares', 'traded_shares']
        }
        
        # Create new DataFrame with standardized column names
        new_df = pd.DataFrame()
        missing_columns = []
        
        # Try to map each required column
        for required_col, variations in column_mappings.items():
            found = False
            for col in df.columns:
                if col in variations:
                    new_df[required_col] = df[col]
                    found = True
                    break
            if not found:
                missing_columns.append(required_col)
        
        # Special handling for missing timestamp - create one from row index
        if 'timestamp' in missing_columns:
            logger.info("No timestamp column found, creating synthetic timestamps from row index")
            new_df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            missing_columns.remove('timestamp')
        
        if missing_columns:
            # Log the actual columns in the CSV for debugging
            logger.info(f"CSV columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}. Found columns: {', '.join(df.columns)}")
        
        # Convert timestamp to datetime (if it's not already)
        if not pd.api.types.is_datetime64_any_dtype(new_df['timestamp']):
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        
        # Remove rows with missing values
        new_df = new_df.dropna()
        
        # Sort by timestamp
        new_df = new_df.sort_values('timestamp')
        
        return new_df
        
    except Exception as e:
        logger.error(f"Error converting CSV: {str(e)}")
        raise ValueError(f"Failed to convert CSV format: {str(e)}")

def validate_and_prepare_dataframe(df):
    """Validate and prepare the DataFrame for analysis."""
    try:
        # Normalize column names
        df = smart_convert_csv(df)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        logger.error(f"Error preparing DataFrame: {str(e)}")
        raise

@csrf_exempt
@require_http_methods(["POST"])
def analyze_strategy_view(request):
    """View to handle strategy analysis requests."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        if 'pineScript' not in request.POST:
            return JsonResponse({'error': 'Pine Script strategy is required'}, status=400)
        
        file = request.FILES['file']
        if not file.name.endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files are supported'}, status=400)
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Log original data sample
        logger.info("Original CSV data sample:")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"First 5 rows:\n{df.head().to_string()}")
        
        # Validate and prepare the DataFrame
        df = validate_and_prepare_dataframe(df)
        
        # Log normalized data sample
        logger.info("\nNormalized data sample:")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"First 5 rows:\n{df.head().to_string()}")
        
        # Get the Pine Script strategy
        pine_script = request.POST['pineScript']
        logger.info(f"\nPine Script strategy:\n{pine_script}")
        
        # Get timeframe (default to 1h if not provided)
        timeframe = request.POST.get('timeframe', '1h')
        logger.info(f"\nTimeframe: {timeframe}")
        
        # Analyze the strategy
        results = analyze_strategy(df, pine_script, timeframe)
        
        # Log analysis results
        logger.info("\nAnalysis results:")
        logger.info(f"Strategy name: {results['strategy_name']}")
        logger.info(f"Total trades: {results['total_trades']}")
        logger.info(f"Win rate: {results['win_rate']:.2%}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Final return: {results['final_return']:.2%}")
        
        # Handle NaN values in the results
        def clean_nan(obj):
            if isinstance(obj, float) and pd.isna(obj):
                return None
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: clean_nan(value) for key, value in obj.items()}
            return obj
        
        # Clean the results before converting to JSON
        cleaned_results = clean_nan(results)
        
        return JsonResponse(cleaned_results, safe=False)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.error(f"Error analyzing strategy: {str(e)}")
        return JsonResponse({'error': f'Error analyzing strategy: {str(e)}'}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_strategy(request):
    if 'script' not in request.FILES or 'name' not in request.POST:
        return JsonResponse({'error': 'Script file and name are required.'}, status=400)
    script = request.FILES['script']
    name = request.POST['name']
    strategy = Strategy.objects.create(name=name, script=script)
    try:
        file_size = f"{(strategy.script.size / 1024):.1f} KB" if strategy.script else "0 KB"
    except:
        file_size = "0 KB"
    return JsonResponse({
        'id': strategy.id, 
        'name': strategy.name, 
        'script': strategy.script.url,
        'uploaded_at': strategy.uploaded_at,
        'file_size': file_size
    })

@csrf_exempt
@require_http_methods(["GET"])
def list_strategies(request):
    strategies = Strategy.objects.all().order_by('-uploaded_at')
    data = []
    for s in strategies:
        try:
            file_size = f"{(s.script.size / 1024):.1f} KB" if s.script else "0 KB"
        except:
            file_size = "0 KB"
        data.append({
            'id': s.id, 
            'name': s.name, 
            'script': s.script.url, 
            'uploaded_at': s.uploaded_at,
            'file_size': file_size
        })
    return JsonResponse(data, safe=False)

@csrf_exempt
@require_http_methods(["POST"])
def upload_datafile(request):
    if 'file' not in request.FILES or 'name' not in request.POST:
        return JsonResponse({'error': 'CSV file and name are required.'}, status=400)
    file = request.FILES['file']
    name = request.POST['name']
    datafile = DataFile.objects.create(name=name, file=file)
    try:
        file_size = f"{(datafile.file.size / 1024):.1f} KB" if datafile.file else "0 KB"
        # Try to count rows in CSV
        import pandas as pd
        df = pd.read_csv(datafile.file.path)
        rows = len(df)
    except:
        file_size = "0 KB"
        rows = 0
    return JsonResponse({
        'id': datafile.id, 
        'name': datafile.name, 
        'file': datafile.file.url,
        'uploaded_at': datafile.uploaded_at,
        'file_size': file_size,
        'rows': rows
    })

@csrf_exempt
@require_http_methods(["GET"])
def list_datafiles(request):
    datafiles = DataFile.objects.all().order_by('-uploaded_at')
    data = []
    for d in datafiles:
        try:
            file_size = f"{(d.file.size / 1024):.1f} KB" if d.file else "0 KB"
            # Try to count rows in CSV
            import pandas as pd
            df = pd.read_csv(d.file.path)
            rows = len(df)
        except:
            file_size = "0 KB"
            rows = 0
        data.append({
            'id': d.id, 
            'name': d.name, 
            'file': d.file.url, 
            'uploaded_at': d.uploaded_at,
            'file_size': file_size,
            'rows': rows
        })
    return JsonResponse(data, safe=False)

@csrf_exempt
@require_http_methods(["POST"])
def run_backtest(request):
    try:
        body = json.loads(request.body)
        strategy_id = body.get('strategy_id')
        datafile_id = body.get('datafile_id')
        logger.info(f"Received run_backtest request: strategy_id={strategy_id}, datafile_id={datafile_id}")
        
        # Check if IDs are provided
        if not strategy_id:
            logger.error("Missing strategy_id")
            return JsonResponse({'error': 'strategy_id is required.'}, status=400)
        if not datafile_id:
            logger.error("Missing datafile_id")
            return JsonResponse({'error': 'datafile_id is required.'}, status=400)
        
        # Check if objects exist
        try:
            strategy = Strategy.objects.get(id=strategy_id)
            logger.info(f"Found strategy: {strategy} (id={strategy.id})")
        except Strategy.DoesNotExist:
            logger.error(f"Strategy with id {strategy_id} not found.")
            return JsonResponse({'error': f'Strategy with id {strategy_id} not found.'}, status=404)
        
        try:
            datafile = DataFile.objects.get(id=datafile_id)
            logger.info(f"Found datafile: {datafile} (id={datafile.id})")
        except DataFile.DoesNotExist:
            logger.error(f"Data file with id {datafile_id} not found.")
            return JsonResponse({'error': f'Data file with id {datafile_id} not found.'}, status=404)

        # Log file paths
        logger.info(f"Strategy script path: {strategy.script.path}")
        logger.info(f"Data file path: {datafile.file.path}")

        # Load the data file as a DataFrame
        import pandas as pd
        data_path = datafile.file.path
        df = pd.read_csv(data_path)
        
        # Use smart converter to handle any CSV format
        try:
            df = smart_convert_csv(df)
            logger.info(f"Successfully converted CSV with columns: {list(df.columns)}")
        except Exception as e:
            return JsonResponse({'error': f'CSV conversion failed: {str(e)}'}, status=400)

        script_path = strategy.script.path

        with open(script_path, 'rb') as f:
           raw = f.read()
           detected = chardet.detect(raw)
           encoding = detected['encoding'] or 'utf-8'

        user_code = raw.decode(encoding, errors='replace')

        # Prepare a comprehensive namespace with all necessary imports
        local_vars = {}
        global_vars = {
            'pd': pd,
            'np': np,
            'floor': floor,
            'ceil': ceil,
            'sqrt': sqrt,
            'log': log,
            'exp': exp,
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'pi': pi,
            'e': math_e,
            'datetime': datetime,
            'timedelta': timedelta,
            'warnings': warnings,
            # Add common pandas functions that users might need
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            'concat': pd.concat,
            'merge': pd.merge,
            'read_csv': pd.read_csv,
            'to_datetime': pd.to_datetime,
            # Add common numpy functions
            'array': np.array,
            'zeros': np.zeros,
            'ones': np.ones,
            'linspace': np.linspace,
            'arange': np.arange,
            'mean': np.mean,
            'std': np.std,
            'sum': np.sum,
            'min': np.min,
            'max': np.max,
            'abs': np.abs,
            'round': round,
            'len': len,
            'range': range,
            'list': list,
            'dict': dict,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Execute the script with comprehensive imports
        try:
            exec(user_code, global_vars, local_vars)
        except NameError as name_error:
            # If there's still a NameError, try to add the missing import
            error_msg = str(name_error)
            if "name 'floor' is not defined" in error_msg:
                # This should not happen now, but just in case
                global_vars['floor'] = floor
                exec(user_code, global_vars, local_vars)
            else:
                raise name_error
        
        if 'run_strategy' not in local_vars:
            # List all functions found in the script
            available_funcs = [k for k, v in local_vars.items() if callable(v)]
            # Try to auto-detect a function that takes a DataFrame as first argument
            candidate_funcs = []
            for k in available_funcs:
                func = local_vars[k]
                try:
                    sig = inspect.signature(func)
                    params = list(sig.parameters.values())
                    if params and (params[0].name == 'df' or params[0].annotation == pd.DataFrame or params[0].kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]):
                        candidate_funcs.append(k)
                except Exception:
                    continue
            if len(candidate_funcs) == 1:
                # Use the single candidate as fallback
                fallback_func = candidate_funcs[0]
                logger.warning(f"run_strategy(df) not found, using fallback function: {fallback_func}")
                run_strategy = local_vars[fallback_func]
                # Warn user in the result
                result = run_strategy(df)
                return JsonResponse({
                    'id': None,
                    'result': result,
                    'warning': f"Your script did not define run_strategy(df). Used fallback: {fallback_func}. Please rename your function to run_strategy for best compatibility."
                })
            elif len(candidate_funcs) > 1:
                error_msg = (
                    "Your script must define a function named run_strategy(df).\n"
                    f"Functions found in your script: {available_funcs}\n"
                    f"Multiple candidate functions found: {candidate_funcs}. Please rename your main function to run_strategy."
                )
                logger.error(error_msg)
                return JsonResponse({'error': error_msg}, status=400)
            else:
                error_msg = (
                    "Your script must define a function named run_strategy(df).\n"
                    f"Functions found in your script: {available_funcs}\n"
                    "No suitable function found that takes a DataFrame as the first argument."
                )
                logger.error(error_msg)
                return JsonResponse({'error': error_msg}, status=400)
        else:
            run_strategy = local_vars['run_strategy']
        # Run the strategy
        result = run_strategy(df)
        # If result is a DataFrame or Series, convert to JSON-serializable format
        if isinstance(result, pd.DataFrame):
            result = result.to_dict(orient='records')
        elif isinstance(result, pd.Series):
            result = result.to_list()
        # Clean NaN values for JSON serialization
        def clean_nan(obj):
            if isinstance(obj, float) and pd.isna(obj):
                return None
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: clean_nan(value) for key, value in obj.items()}
            return obj
        result = clean_nan(result)
        # Store the result
        backtest_result = BacktestResult.objects.create(strategy=strategy, datafile=datafile, result=result)
        logger.info(f"Backtest completed and stored with id {backtest_result.id}")
        return JsonResponse({'id': backtest_result.id, 'result': result})
    except json.JSONDecodeError:
        logger.error('Invalid JSON in request body.')
        return JsonResponse({'error': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Exception in run_backtest: {str(e)}\n{tb}")
        return JsonResponse({'error': str(e), 'traceback': tb}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def list_backtest_results(request):
    # Get pagination parameters
    page = int(request.GET.get('page', 1))
    page_size = int(request.GET.get('page_size', 10))
    
    # Get all results ordered by creation date
    all_results = BacktestResult.objects.all().order_by('-created_at')
    
    # Calculate pagination
    total_count = all_results.count()
    total_pages = (total_count + page_size - 1) // page_size
    
    # Get results for current page
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    results = all_results[start_index:end_index]
    
    data = []
    for r in results:
        # Flatten the result structure to match frontend expectations
        backtest_data = {
            'id': r.id,
            'strategy': r.strategy.name,
            'datafile': r.datafile.name,
            'created_at': r.created_at,
            'date': r.created_at.strftime('%Y-%m-%d %H:%M:%S'),  # Add formatted date for frontend
        }
        
        # Extract summary and trades from the result
        if isinstance(r.result, dict):
            if 'summary' in r.result:
                backtest_data['summary'] = r.result['summary']
            if 'trades' in r.result:
                backtest_data['trades'] = r.result['trades']
            if 'currency' in r.result:
                backtest_data['currency'] = r.result['currency']
            # Include the full result as well for backward compatibility
            backtest_data['result'] = r.result
        else:
            # If result is not a dict, include it as is
            backtest_data['result'] = r.result
            
        data.append(backtest_data)
    
    # Return paginated response
    return JsonResponse({
        'results': data,
        'pagination': {
            'current_page': page,
            'page_size': page_size,
            'total_count': total_count,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_previous': page > 1
        }
    }, safe=False)

# Delete endpoints
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_strategy(request, strategy_id):
    try:
        strategy = Strategy.objects.get(id=strategy_id)
        strategy.delete()
        return JsonResponse({'message': 'Strategy deleted successfully'}, status=200)
    except Strategy.DoesNotExist:
        return JsonResponse({'error': 'Strategy not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_datafile(request, datafile_id):
    try:
        datafile = DataFile.objects.get(id=datafile_id)
        datafile.delete()
        return JsonResponse({'message': 'Data file deleted successfully'}, status=200)
    except DataFile.DoesNotExist:
        return JsonResponse({'error': 'Data file not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_backtest(request, backtest_id):
    try:
        backtest = BacktestResult.objects.get(id=backtest_id)
        backtest.delete()
        return JsonResponse({'message': 'Backtest result deleted successfully'}, status=200)
    except BacktestResult.DoesNotExist:
        return JsonResponse({'error': 'Backtest result not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Update endpoints
@csrf_exempt
@require_http_methods(["PUT", "PATCH", "POST"])
def update_strategy(request, strategy_id):
    try:
        logger.info(f"Updating strategy {strategy_id}")
        logger.info(f"Request POST data: {request.POST}")
        logger.info(f"Request FILES: {request.FILES}")
        
        strategy = Strategy.objects.get(id=strategy_id)
        logger.info(f"Found strategy: {strategy.name}")
        
        # Handle FormData instead of JSON
        if 'name' in request.POST:
            old_name = strategy.name
            strategy.name = request.POST['name']
            logger.info(f"Updating name from '{old_name}' to '{strategy.name}'")
        
        if 'script' in request.FILES:
            logger.info(f"Updating script file: {request.FILES['script'].name}")
            strategy.script = request.FILES['script']
        
        strategy.save()
        logger.info(f"Strategy saved successfully")
        
        try:
            file_size = f"{(strategy.script.size / 1024):.1f} KB" if strategy.script else "0 KB"
        except:
            file_size = "0 KB"
        
        response_data = {
            'id': strategy.id,
            'name': strategy.name,
            'script': strategy.script.url,
            'uploaded_at': strategy.uploaded_at,
            'file_size': file_size
        }
        logger.info(f"Returning response: {response_data}")
        return JsonResponse(response_data)
    except Strategy.DoesNotExist:
        logger.error(f"Strategy {strategy_id} not found")
        return JsonResponse({'error': 'Strategy not found'}, status=404)
    except Exception as e:
        logger.error(f"Error updating strategy: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["PUT", "PATCH", "POST"])
def update_datafile(request, datafile_id):
    try:
        print(f"[update_datafile] POST: {request.POST}")
        print(f"[update_datafile] FILES: {request.FILES}")
        datafile = DataFile.objects.get(id=datafile_id)
        
        # Handle FormData instead of JSON
        if 'name' in request.POST:
            datafile.name = request.POST['name']
        
        if 'file' in request.FILES:
            datafile.file = request.FILES['file']
        
        datafile.save()
        
        try:
            file_size = f"{(datafile.file.size / 1024):.1f} KB" if datafile.file else "0 KB"
            # Try to count rows in CSV
            import pandas as pd
            df = pd.read_csv(datafile.file.path)
            rows = len(df)
        except:
            file_size = "0 KB"
            rows = 0
        
        return JsonResponse({
            'id': datafile.id,
            'name': datafile.name,
            'file': datafile.file.url,
            'uploaded_at': datafile.uploaded_at,
            'file_size': file_size,
            'rows': rows
        })
    except DataFile.DoesNotExist:
        return JsonResponse({'error': 'Data file not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def list_stock_symbols(request):
    data_dir = os.path.join(settings.MEDIA_ROOT, 'all_stock_data')
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    symbols = [os.path.splitext(f)[0] for f in files]
    return JsonResponse({'symbols': symbols})

@csrf_exempt
def get_stock_data(request):
    symbol = request.GET.get('symbol')
    start = request.GET.get('start')
    end = request.GET.get('end')
    if not symbol:
        return JsonResponse({'error': 'symbol required'}, status=400)
    data_dir = os.path.join(settings.MEDIA_ROOT, 'all_stock_data')
    file_path = os.path.join(data_dir, f'{symbol}.csv')
    if not os.path.exists(file_path):
        return JsonResponse({'error': 'file not found'}, status=404)
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
    # Try to parse date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        if start:
            df = df[df[date_col] >= pd.to_datetime(start)]
        if end:
            df = df[df[date_col] <= pd.to_datetime(end)]
    data = df.head(500).to_dict(orient='records')  # limit for performance
    return JsonResponse({'data': data, 'columns': list(df.columns)})

@csrf_exempt
@require_http_methods(["GET"])
def compatible_stocks(request):
    import os
    import time
    import inspect
    import pandas as pd
    import numpy as np
    from math import floor, ceil, sqrt, log, exp, sin, cos, tan, pi, e as math_e
    from datetime import datetime, timedelta
    import traceback

    try:
        strategy_id = request.GET.get('strategy_id')
        show_errors = request.GET.get('show_errors') == '1'

        if not strategy_id:
            return JsonResponse({'status': 'error', 'error': 'strategy_id required'}, status=400)

        try:
            strategy = Strategy.objects.get(id=int(strategy_id))
        except (Strategy.DoesNotExist, ValueError, TypeError):
            return JsonResponse({'status': 'error', 'error': 'Strategy not found'}, status=404)

        with open(strategy.script.path, 'r', encoding='utf-8', errors='replace') as f:
            user_code = f.read()

        compatible = []
        errors = []
        latest_dates = {}

        data_dir = os.path.join(settings.MEDIA_ROOT, 'all_stock_data')
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        files = files[:10]  # limit to avoid long load

        for file in files:
            symbol = os.path.splitext(file)[0]
            file_path = os.path.join(data_dir, file)
            print(f"🔍 Processing: {symbol}")

            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                df = smart_convert_csv(df)

                # Normalize date column
                if 'date' not in df.columns and 'timestamp' in df.columns:
                    df['date'] = df['timestamp']
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                        latest_date = df['date'].max().strftime('%Y-%m-%d')
                    except Exception:
                        latest_date = None
                else:
                    latest_date = None

                local_vars = {}
                global_vars = {
                    'pd': pd, 'np': np, 'floor': floor, 'ceil': ceil, 'sqrt': sqrt, 'log': log, 'exp': exp,
                    'sin': sin, 'cos': cos, 'tan': tan, 'pi': pi, 'e': math_e, 'datetime': datetime, 'timedelta': timedelta,
                    'warnings': __import__('warnings'), 'DataFrame': pd.DataFrame, 'Series': pd.Series, 'concat': pd.concat, 'merge': pd.merge,
                    'read_csv': pd.read_csv, 'to_datetime': pd.to_datetime, 'array': np.array, 'zeros': np.zeros, 'ones': np.ones,
                    'linspace': np.linspace, 'arange': np.arange, 'mean': np.mean, 'std': np.std, 'sum': np.sum, 'min': np.min,
                    'max': np.max, 'abs': abs, 'round': round, 'len': len, 'range': range, 'list': list, 'dict': dict,
                    'str': str, 'int': int, 'float': float, 'bool': bool, 'True': True, 'False': False, 'None': None,
                }

                exec(user_code, global_vars, local_vars)

                # Resolve run_strategy
                if 'run_strategy' in local_vars:
                    run_strategy = local_vars['run_strategy']
                else:
                    funcs = [f for f in local_vars if callable(local_vars[f])]
                    run_strategy = None
                    for name in funcs:
                        try:
                            sig = inspect.signature(local_vars[name])
                            if sig.parameters and list(sig.parameters)[0] in ['df']:
                                run_strategy = local_vars[name]
                                break
                        except Exception:
                            continue
                    if not run_strategy:
                        errors.append({'symbol': symbol, 'error': 'No suitable run_strategy function found'})
                        continue

                # Run strategy
                try:
                    start = time.time()
                    result = run_strategy(df)
                    elapsed = time.time() - start
                    if elapsed > 10:
                        errors.append({'symbol': symbol, 'error': f"Timeout ({elapsed:.2f}s)"})
                    else:
                        compatible.append(symbol)
                        if latest_date:
                            latest_dates[symbol] = latest_date
                except Exception as e:
                    errors.append({'symbol': symbol, 'error': str(e), 'trace': traceback.format_exc()})

            except Exception as e:
                errors.append({'symbol': symbol, 'error': str(e), 'trace': traceback.format_exc()})

        return JsonResponse({
            'status': 'ok',
            'strategy_id': strategy_id,
            'compatible_stocks': compatible,
            'latest_dates': latest_dates,
            'error_count': len(errors),
            'errors': errors if show_errors else []
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=500)
