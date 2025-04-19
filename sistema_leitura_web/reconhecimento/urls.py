from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('capturar/', views.capturar_placa, name='capturar'),
    path('reconhecer/', views.reconhecer_placa, name='reconhecer'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
