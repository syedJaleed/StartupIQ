import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { MainDashComponent } from './main-dash/main-dash.component';

const routes: Routes = [
  {
    path: '',
    component: DashboardComponent,
    children: [{ path: '', component: MainDashComponent }],
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class DashboardRoutingModule {}
