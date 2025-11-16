import { NavLink } from 'react-router-dom';
import './NavBar.css';

const NavBar = () => {
  return (
    <nav className="navbar">
      <div className="navbar__brand">Article summarisation</div>
      <div className="navbar__links">
        <NavLink to="/" end className="navbar__link">
          Dashboard
        </NavLink>
        <NavLink to="/demo" className="navbar__link">
          Demo
        </NavLink>
      </div>
    </nav>
  );
};

export default NavBar;
