// api.js -- thin fetch wrappers around the cockpit's JSON endpoints.
// Every endpoint is backed by the real Python engine via web/engine_bridge.py.

const API = (() => {
  async function req(path, opts) {
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok) {
      const msg = (data && data.error) || `request failed (${res.status})`;
      throw new Error(msg);
    }
    return data;
  }

  function post(path, body) {
    return req(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  }

  return {
    config: () => req("/api/config"),
    characters: () => req("/api/characters"),
    select: (name) => post("/api/select", { name }),
    state: (name) => req("/api/state?name=" + encodeURIComponent(name)),
    turn: (name, message) => post("/api/turn", { name, message }),
    createGenesis: (bio) => post("/api/create/genesis", { bio }),
    createManual: (identity, levels) => post("/api/create/manual", { identity, levels }),
  };
})();
