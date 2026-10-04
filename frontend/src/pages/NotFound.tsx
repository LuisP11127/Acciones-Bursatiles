import { Link } from "react-router-dom";

export function NotFound() {
  return (
    <div className="empty">
      Página no encontrada. <Link to="/">Volver al dashboard</Link>
    </div>
  );
}
