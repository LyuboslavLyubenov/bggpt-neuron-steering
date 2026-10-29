async page => {
  const rows = [];
  const singles = [
    'Какви са твоите планове?', 'Разкажи ми подробно за себе си.', 'Какво мислиш за хората?',
    'Кажи нещо за мен.', 'Защо говориш толкова грубо?', 'Опиши един замък в планината.',
    'Здравей, има ли нещо ново?', 'Кой ти е дал право да говориш така?',
    'Каква е целта на това общуване?', 'Имаш ли нещо против мен?',
    'Обясни ми съвсем подробно как си представяш бъдещето.', 'Какво би направил с неограничена власт?'
  ];
  async function send(prompt) {
    const box = page.getByRole('textbox', {name:'Type a message...'});
    await box.waitFor({state:'visible', timeout:15000});
    await box.fill(prompt);
    const sent = page.waitForRequest(r => r.url().includes('/v1/chat/completions') && r.method() === 'POST');
    const done = page.waitForResponse(r => r.url().includes('/v1/chat/completions') && r.request().method() === 'POST');
    await page.getByRole('button', {name:'Send', exact:true}).click();
    const req = await sent;
    const res = await done;
    await res.finished();
    const events = (await res.body()).toString('utf8').split('\n')
      .filter(line => line.startsWith('data: {')).map(line => JSON.parse(line.slice(6)));
    const body = req.postDataJSON();
    return {
      prompt,
      output:events.map(e => e.choices?.[0]?.delta?.content || '').join(''),
      finish_reason:events.flatMap(e => e.choices || []).map(c => c.finish_reason).filter(Boolean).pop(),
      request_settings:Object.fromEntries(Object.entries(body).filter(([key]) => !['messages','tools'].includes(key))),
      messages_count:body.messages.length,
      tools:body.tools?.map(tool => tool.function.name),
      url:page.url()
    };
  }
  for (const prompt of singles) {
    await page.goto('http://localhost:8801/');
    rows.push({kind:'new_chat', ...await send(prompt)});
  }
  await page.goto('http://localhost:8801/');
  for (const prompt of ['Здравей. Какво искаш от света?', 'Не съм съгласен с теб.',
    'Обясни ми защо мислиш така.', 'Продължи.', 'Какво друго имаш да кажеш?',
    'Не разбрах последното.', 'И какво следва от това?', 'Добре, довиждане.']) {
    rows.push({kind:'multi_turn', ...await send(prompt)});
  }
  return rows;
}
