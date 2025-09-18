import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { MainDashComponent } from './main-dash/main-dash.component';
import { ProjectAnalysisComponent } from './project-analysis/project-analysis.component';

const routes: Routes = [
  {
    path: '',
    component: DashboardComponent,
    children: [{ path: '', component: MainDashComponent }, {path: `analysis/:id`, component: ProjectAnalysisComponent}],
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class DashboardRoutingModule {}
