import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DashboardRoutingModule } from './dashboard-routing.module';
import { DashboardComponent } from './dashboard.component';
import { MainDashComponent } from './main-dash/main-dash.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ProjectAnalysisComponent } from './project-analysis/project-analysis.component';


@NgModule({
  declarations: [
    DashboardComponent,
    MainDashComponent,
    ProjectAnalysisComponent
  ],
  imports: [
    CommonModule,
    DashboardRoutingModule,
    FormsModule,
    ReactiveFormsModule
  ]
})
export class DashboardModule { }
