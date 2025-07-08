# core/urls.py

from django.urls import path
from .views import backtest_view, list_stock_symbols, get_stock_data
from . import views

urlpatterns = [
    path("backtest/", backtest_view, name="backtest"),
    path('analyze-strategy/', views.analyze_strategy_view, name='analyze_strategy'),
    path('upload-strategy/', views.upload_strategy, name='upload_strategy'),
    path('list-strategies/', views.list_strategies, name='list_strategies'),
    path('upload-datafile/', views.upload_datafile, name='upload_datafile'),
    path('list-datafiles/', views.list_datafiles, name='list_datafiles'),
    path('run-backtest/', views.run_backtest, name='run_backtest'),
    path('list-backtest-results/', views.list_backtest_results, name='list_backtest_results'),
    
    # Delete endpoints
    path('delete-strategy/<int:strategy_id>/', views.delete_strategy, name='delete_strategy'),
    path('delete-datafile/<int:datafile_id>/', views.delete_datafile, name='delete_datafile'),
    path('delete-backtest/<int:backtest_id>/', views.delete_backtest, name='delete_backtest'),
    
    # Update endpoints
    path('update-strategy/<int:strategy_id>/', views.update_strategy, name='update_strategy'),
    path('update-datafile/<int:datafile_id>/', views.update_datafile, name='update_datafile'),
    path('list-stock-symbols/', list_stock_symbols, name='list_stock_symbols'),
    path('get-stock-data/', get_stock_data, name='get_stock_data'),
    path('compatible-stocks/', views.compatible_stocks, name='compatible_stocks'),
]
