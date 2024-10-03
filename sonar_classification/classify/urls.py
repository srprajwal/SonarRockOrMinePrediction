from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_sonar, name='predict_sonar'),
]
