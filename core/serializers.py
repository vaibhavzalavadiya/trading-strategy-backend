from rest_framework import serializers
from .models import Strategy, DataFile, BacktestResult

class StrategySerializer(serializers.ModelSerializer):
    class Meta:
        model = Strategy
        fields = ['id', 'name', 'script', 'uploaded_at']

class DataFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataFile
        fields = ['id', 'name', 'file', 'uploaded_at']

class BacktestResultSerializer(serializers.ModelSerializer):
    strategy = StrategySerializer(read_only=True)
    datafile = DataFileSerializer(read_only=True)
    strategy_id = serializers.PrimaryKeyRelatedField(queryset=Strategy.objects.all(), source='strategy', write_only=True)
    datafile_id = serializers.PrimaryKeyRelatedField(queryset=DataFile.objects.all(), source='datafile', write_only=True)

    class Meta:
        model = BacktestResult
        fields = ['id', 'strategy', 'datafile', 'strategy_id', 'datafile_id', 'result', 'created_at'] 