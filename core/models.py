from django.db import models

# Create your models here.

class Strategy(models.Model):
    name = models.CharField(max_length=255)
    script = models.FileField(upload_to='strategies/')  # Python script file
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class DataFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datafiles/')  # CSV file
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class BacktestResult(models.Model):
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    datafile = models.ForeignKey(DataFile, on_delete=models.CASCADE)
    result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Backtest: {self.strategy.name} on {self.datafile.name} at {self.created_at}"
