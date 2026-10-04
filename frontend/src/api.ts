import { useEffect, useState } from "react";

const cache = new Map<string, Promise<unknown>>();

/** Carga un JSON de public/data (generado por GitHub Actions). */
export function loadJson<T>(path: string): Promise<T> {
  let promesa = cache.get(path);
  if (!promesa) {
    const url = `${import.meta.env.BASE_URL}data/${path}`;
    promesa = fetch(url, { cache: "no-cache" }).then(async (r) => {
      if (!r.ok) throw new Error(r.status === 404 ? "Aún no hay datos publicados" : `Error ${r.status} al cargar ${path}`);
      return r.json();
    });
    promesa.catch(() => cache.delete(path));
    cache.set(path, promesa);
  }
  return promesa as Promise<T>;
}

export interface Loadable<T> {
  data?: T;
  error?: string;
  loading: boolean;
}

export function useJson<T>(path: string | null): Loadable<T> {
  const [estado, setEstado] = useState<Loadable<T>>({ loading: Boolean(path) });
  useEffect(() => {
    if (!path) return;
    let vivo = true;
    setEstado((e) => ({ ...e, loading: true, error: undefined }));
    loadJson<T>(path)
      .then((data) => vivo && setEstado({ data, loading: false }))
      .catch((err: Error) => vivo && setEstado({ error: err.message, loading: false }));
    return () => {
      vivo = false;
    };
  }, [path]);
  return estado;
}
