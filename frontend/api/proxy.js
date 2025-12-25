export default async function handler(req, res) {
  try {
    const base = process.env.API_BASE_URL;
    if (!base) {
      res.status(500).json({ error: 'API_BASE_URL env is not set' });
      return;
    }
    const url = new URL(req.url, base);
    const target = base.replace(/\/$/, '') + req.url;
    const init = {
      method: req.method,
      headers: Object.fromEntries(Object.entries(req.headers).filter(([k]) => !['host'].includes(k.toLowerCase()))),
      body: req.method === 'GET' || req.method === 'HEAD' ? undefined : req,
    };
    const response = await fetch(target, init);
    const buf = Buffer.from(await response.arrayBuffer());
    for (const [k, v] of response.headers) res.setHeader(k, v);
    res.status(response.status).send(buf);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
}
